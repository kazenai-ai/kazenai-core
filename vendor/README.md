# Vendored wheels (CI)

Pinned wheels for hermetic CI. **`kazen-event-schema` is on PyPI** (`0.6.2`);
**`kazenai-contracts` is not** — keep that wheel here until it is published.

| Wheel | Purpose |
|-------|---------|
| `kazen_event_schema-0.6.2-py3-none-any.whl` | Optional pin; also available via `pip install kazen-event-schema` |
| `kazenai_contracts-0.1.0-py3-none-any.whl` | Provides `kazenai_contracts` for principal/auth tests (not on PyPI) |

Refresh:

```bash
# schema (or: pip download kazen-event-schema==0.6.2)
cd ../kazen-event-schema && python -m build
cp dist/kazen_event_schema-0.6.2-py3-none-any.whl ../kazenai-core/vendor/

# contracts
cd ../kazenai-contracts/sdks/python && python -m build
cp dist/kazenai_contracts-0.1.0-py3-none-any.whl ../kazenai-core/vendor/
```
