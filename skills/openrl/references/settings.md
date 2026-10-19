# openrl.* settings

The settings are not listed here, on purpose. The server you are connected to
describes them in one place: the JSON Schema `components.schemas.OpenRLSettings`
in its OpenAPI document. `/docs` renders the same schema for a human, and
`get_server_capabilities` points at it as `openrl.settings_schema_url`. Read it
from the server, so the answer is for the deployment you are using and its release:

```bash
curl -s $OPENRL/openapi.json | jq '.components.schemas.OpenRLSettings.properties'
```

## Reading the schema

- `properties` are the settings, keyed by their wire name, `openrl.<name>`.
- `type` is `integer`, `boolean` or `string`; `minimum` and `maximum` bound an integer;
  `enum` lists the allowed strings.
- `anyOf: [X, {type: null}]` with `default: null` is a setting that is unset unless you pass it.
- `description` says what the setting does and which fine-tuning type it applies to.
- `additionalProperties: false`: an `openrl.` key that is not in `properties` is a 400.

## Passing settings

Settings travel in `user_metadata`, on `create_session` (for every model in the session)
or on `create_model` (for that model; overrides the session), or in the client's
environment as `TINKER_TAGS=openrl.key=value,openrl.other=value`. Values are strings on
the wire: `"true"`, `"8"`. A bad value or an unknown key is a 400 that names the setting
and says what it should be.

```python
client = tinker.ServiceClient(base_url=..., user_metadata={
  "openrl.trainer_dtype": "bfloat16",  # required on every model, no default
  "openrl.sampler_dtype": "bfloat16",  # required; must equal trainer_dtype for now
  "openrl.fine_tuning_type": "full",
})
```

`openrl.trainer_dtype` and `openrl.sampler_dtype` are required: a model without them is a
400. On a model restored from a checkpoint they are required too and must match the dtype
the checkpoint was saved in. The schema's descriptions say which trainer supports which
dtype.

Which settings a server accepts depends on how it runs workers: several are refused
outside the Kubernetes deployment (`openrl.worker_launcher` in capabilities is not
`scheduler`), and the 400 says so. The decision guide for bigger models, which turns
on `openrl.exclusive`, `openrl.trainer_backend` and `openrl.trainer_gpus`, is in
`lifecycle.md`.
