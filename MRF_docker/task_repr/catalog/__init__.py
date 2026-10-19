"""静态组件库的加载与查询。"""

from .component_catalog import ComponentCatalog, default_component_catalog, load_component_catalog

__all__ = ["ComponentCatalog", "default_component_catalog", "load_component_catalog"]
