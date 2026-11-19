# 07_newspaper_map.R -- where the newspapers were printed, sized by articles
# (1740-1848, same data as 06_gam.R). Writes figs/newspaper_map.pdf.

library(here)
source(here("src", "config.R"))
library(purrr)
library(jsonlite)
library(ggrepel)
library(mapdata)

papers <- fromJSON(here("resources", "newspapers.json"))
papers <- tibble(newspaper = names(papers),
                 place  = map_chr(papers, "place"),
                 region = map_chr(papers, "region"))

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

counts <- load_gam_data() |> count(newspaper, name = "articles")

dots <- papers |>
  left_join(counts, by = "newspaper") |>
  mutate(articles = coalesce(articles, 0L)) |>
  group_by(place) |>
  summarise(titles = n(), articles = sum(articles), .groups = "drop") |>
  left_join(towns, by = "place") |>
  mutate(lab = sprintf("%s (%d)", place, titles))
stopifnot(!anyNA(dots$lon))   # every place needs coordinates

land <- map_data("worldHires", region = c("Denmark", "Norway", "Sweden", "Germany"))

p <- ggplot() +
  geom_polygon(data = land, aes(long, lat, group = group,
                               fill = region %in% c("Denmark", "Norway")),
               colour = "grey70", linewidth = .2) +
  scale_fill_manual(values = c(`TRUE` = "grey85", `FALSE` = "grey96"), guide = "none") +
  geom_point(data = dots, aes(lon, lat, size = articles),
             colour = "#2166ac", alpha = .7) +
  geom_text_repel(data = dots, aes(lon, lat, label = lab), size = 4.5,
                  min.segment.length = 0, seed = 1) +
  scale_size_area(max_size = 18, labels = scales::label_comma(),
                  breaks = c(1e4, 1e5, 1e6), name = "Articles") +
  coord_quickmap(xlim = c(4, 16), ylim = c(54.4, 64)) +
  labs(x = NULL, y = NULL) +
  theme_void(base_size = 16) +
  theme(legend.position = "bottom")

ggsave(file.path(FIGS, "newspaper_map.pdf"), p, width = 7, height = 9)
print(as.data.frame(dots |> arrange(desc(articles)) |> select(place, titles, articles)))
