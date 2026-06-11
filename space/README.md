# Hugging Face Space Template

Copy this folder into a Hugging Face Space when you want browser upload:

- User uploads `.pdf`, `.docx`, `.md`, or `.txt`
- Space loads `IS_IDENTIFIER_MODEL`
- Space returns the Paso 1 Excel (sheets: `segments` / `schema` / `summary`;
  candidates + review flags, no taxonomic coding)

Before publishing:

1. Confirm the model id in `app.py`.
2. Confirm the GitHub URL in `requirements.txt`.
3. Add this Space secret or variable if you want to override the default:

```text
IS_IDENTIFIER_MODEL=bravo-pena/is-identifier-1.0
```

No training data is required by the Space.
