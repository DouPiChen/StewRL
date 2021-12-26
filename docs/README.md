# Eden中文文档

这是一种可配置的生存游戏环境Eden的中文文档网站。

## site.pages

<!-- prettier-ignore-start -->

| source          | link                                                           |
| --------------- | -------------------------------------------------------------- |
{% for page in site.pages -%}
| {{ page.path }} | [{{ page.url | relative_url }}]({{ page.url | relative_url }}) |
{% endfor %}

<!-- prettier-ignore-end -->


## Local debug

```sh
make
make server
```

## The license




