# Vendored wheels (CI)

These packages are not published to PyPI. CI installs the pinned wheels here
before `pip install -e ".[dev]"`.

| Wheel | Purpose |
|-------|---------|
| `kazen_event_schema-0.6.0-py3-none-any.whl` | Satisfies `kazen-event-schema>=0.6.0,<0.7` |
| `kazenai_contracts-0.1.0-py3-none-any.whl` | Provides `kazenai_contracts` for principal/auth tests |

Refresh:

```bash
# schema
cd ../kazen-event-schema && python -m build
cp dist/kazen_event_schema-0.6.0-py3-none-any.whl ../kazenai-core/vendor/

# contracts
cd ../kazenai-contracts/sdks/python && python -m build
cp dist/kazenai_contracts-0.1.0-py3-none-any.whl ../kazenai-core/vendor/
```
