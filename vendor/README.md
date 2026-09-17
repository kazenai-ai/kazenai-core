# Vendored wheels (CI)

`kazen-event-schema` is not published to PyPI. CI installs the pinned wheel here until a
public or private index serves `>=0.6.0,<0.7`.

Refresh:
```bash
cd ../kazen-event-schema && python -m build
cp dist/kazen_event_schema-0.6.0-py3-none-any.whl ../kazenai-core/vendor/
```
