# 07_newspaper_map.R -- where the newspapers were printed, sized by articles
# (whole corpus, 1666-1849, no paratext, OCR >= 0.5). Writes figs/newspaper_map.pdf.

library(here)
source(here("src", "config.R"))
library(purrr)
library(jsonlite)
library(ggrepel)
library(mapdata)

papers <- fromJSON(here("resources", "newspapers.json"))
papers <- tibble(newspaper = names(papers),
                 place  = purrr::map_chr(papers, "place"),
                 region = purrr::map_chr(papers, "region"))

# printing towns, lon/lat
towns <- tribble(
  ~place,             ~lon,   ~lat,
  "K\u00f8benhavn",        12.57,  55.68,
  "Aalborg",           9.92,  57.05,
  "Aarhus",           10.21,  56.16,
  "Odense",           10.39,  55.40,
  "Ribe",              8.76,  55.33,
  "Viborg",            9.40,  56.45,
  "Thisted",           8.69,  56.96,
  "Slagelse",         11.35,  55.40,
  "Nyk\u00f8bing Falster", 11.87,  54.77,
  "Christiania",      10.75,  59.91,
  "Bergen",            5.32,  60.39,
  "Trondheim",        10.40,  63.43)

counts <- read_parquet(file.path(DATA, "eno", "pwa.parquet"),
                       col_select = c("newspaper", "predicted_category", "pwa")) |>
  filter(predicted_category != "Paratext", pwa >= 0.5) |>
  count(newspaper, name = "articles")

dots <- papers |>
  left_join(counts, by = "newspaper") |>
  mutate(articles = coalesce(articles, 0L)) |>
  group_by(place) |>
  summarise(titles = n(), articles = sum(articles), .groups = "drop") |>
  left_join(towns, by = "place") |>
  mutate(lab = sprintf("%s (%d)", place, titles))
stopifnot(!anyNA(dots$lon))   # every place needs coordinates

land <- map_data("worldHires", region = c("Denmark", "Norway", "Sweden", "Germany"))

# the realm under the Danish crown 1660-1814: Denmark, Norway, and the duchies
# of Schleswig and Holstein. The duchies are cut out of modern Germany with a
# rough polygon along the Elbe and Trave -- approximate, not a surveyed border.
duchies <- list(x = c(7.8, 8.9, 9.5, 10.0, 10.6, 10.95, 10.9, 11.4, 11.4, 7.8),
                y = c(54.0, 53.88, 53.62, 53.53, 53.37, 53.6, 53.95, 54.4, 55.2, 55.2))
sh <- land |>
  filter(region == "Germany") |>
  split(~ group) |>
  purrr::map(\(g) polyclip::polyclip(list(x = g$long, y = g$lat), duchies, op = "intersect")) |>
  list_flatten() |>
  imap(\(q, i) tibble(long = q$x, lat = q$y, group = paste0("sh", i), realm = TRUE)) |>
  list_rbind()
land <- land |>
  mutate(realm = region %in% c("Denmark", "Norway"), group = as.character(group)) |>
  bind_rows(sh) |>
  mutate(modern = coalesce(region %in% c("Denmark", "Norway"), FALSE))

# dot texture for the realm: a regular grid, kept where it falls inside a
# realm polygon (the duchies' polygons are cut out of Germany, so inside one
# means inside the realm)
grid <- expand_grid(long = seq(4, 16, by = .12), lat = seq(53.3, 64, by = .07))
inside <- land |>
  filter(realm) |>
  split(~ group) |>
  purrr::map(\(g) polyclip::pointinpolygon(list(x = grid$long, y = grid$lat),
                                           list(x = g$long, y = g$lat)) != 0) |>
  purrr::reduce(`|`)
texture <- grid[inside, ]

p <- ggplot() +
  # grey = present-day Denmark and Norway; dots = the realm 1660-1814
  geom_polygon(data = land, aes(long, lat, group = group, fill = modern),
               colour = "grey70", linewidth = .2) +
  scale_fill_manual(values = c(`TRUE` = "grey85", `FALSE` = "grey96"), guide = "none") +
  geom_point(data = texture, aes(long, lat), size = .25, colour = "grey30",
             shape = 16) +
  geom_point(data = dots, aes(lon, lat, size = articles),
             colour = "#2166ac", alpha = .7) +
  geom_text_repel(data = dots, aes(lon, lat, label = lab), size = 4.5,
                  min.segment.length = 0, seed = 1) +
  scale_size_area(max_size = 18, labels = scales::label_comma(),
                  breaks = c(1e4, 1e5, 1e6), name = "Articles") +
  coord_quickmap(xlim = c(4, 16), ylim = c(53.3, 64)) +
  labs(x = NULL, y = NULL) +
  theme_void(base_size = 16) +
  theme(legend.position = "right")

ggsave(file.path(FIGS, "newspaper_map.pdf"), p, width = 8.5, height = 9)
print(as.data.frame(dots |> arrange(desc(articles)) |> select(place, titles, articles)))

# --- second map: same towns over the Viabundus road network --------------------
# Viabundus v2 (Zenodo 16611998, CC BY 4.0), data/viabundus/. Roads and ferries
# in Denmark (DMK), Schleswig (SLE) and Holstein (HOL), dropping segments that
# end before 1666. No Norwegian roads in Viabundus, so this map stops at Skagen.
vb_file <- file.path(DATA, "viabundus", "Viabundus-2-edges.geojson")
if (file.exists(vb_file)) {
  vb <- jsonlite::fromJSON(vb_file, simplifyVector = FALSE)$features
  keep <- purrr::keep(vb, \(f) {
    p <- f$properties
    p$section %in% c("DMK", "SLE", "HOL") && p$type %in% c("land", "ferry") &&
      (is.null(p$toyear) || p$toyear >= 1666)
  })
  roads <- purrr::imap(keep, \(f, i) {
    g <- f$geometry
    lines <- if (g$type == "LineString") list(g$coordinates) else g$coordinates
    purrr::imap(lines, \(l, j) tibble(
      long = purrr::map_dbl(l, 1), lat = purrr::map_dbl(l, 2),
      id = paste(i, j), type = f$properties$type,
      zoom = f$properties$zoomlevel))
  }) |> list_flatten() |> list_rbind() |>
    # Viabundus' Denmark section includes Scania, Swedish after 1658
    group_by(id) |>
    filter(!(mean(long) > 12.7 & mean(lat) > 55.33)) |>
    ungroup()

  dk_towns <- dots |> filter(lat < 58)
  p2 <- ggplot() +
    geom_polygon(data = land, aes(long, lat, group = group, fill = modern),
                 colour = "grey70", linewidth = .2) +
    scale_fill_manual(values = c(`TRUE` = "grey85", `FALSE` = "grey96"), guide = "none") +
    geom_path(data = roads, aes(long, lat, group = id, linewidth = zoom,
                                linetype = type), colour = "#7f5a3c") +
    scale_linewidth(range = c(.6, .15), breaks = 1:4, guide = "none") +
    scale_linetype_manual(values = c(land = "solid", ferry = "22"),
                          labels = c(land = "road", ferry = "ferry"), name = NULL) +
    geom_point(data = dk_towns, aes(lon, lat, size = articles),
               colour = "#2166ac", alpha = .75) +
    geom_text_repel(data = dk_towns, aes(lon, lat, label = lab), size = 4.5,
                    min.segment.length = 0, seed = 1, bg.colour = "white", bg.r = .12) +
    scale_size_area(max_size = 18, labels = scales::label_comma(),
                    breaks = c(1e4, 1e5, 1e6), name = "Articles") +
    coord_quickmap(xlim = c(7.9, 15.3), ylim = c(53.4, 57.8)) +
    theme_void(base_size = 16) +
    theme(legend.position = "right")
  ggsave(file.path(FIGS, "newspaper_map_roads.pdf"), p2, width = 9, height = 8)
}
