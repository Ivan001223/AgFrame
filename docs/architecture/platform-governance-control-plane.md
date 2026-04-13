# Platform Governance Control Plane

The governance control plane is responsible for lifecycle authorization and for
maintaining a single authoritative write path for run status changes.

## Single authoritative write path

The intended single authoritative write path for governance decisions is
`app/platform/governance/service.py`.

Current control-plane integrations route harness lifecycle decisions through:

- `app/platform/governance/service.py`
- `app/platform/governance/lifecycle.py`
- `app/harness/runtime/run_service.py`

This keeps runtime execution separate from governance authorization while the
monolith is still being decomposed.
