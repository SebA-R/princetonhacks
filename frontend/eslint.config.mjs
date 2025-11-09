import next from "eslint-config-next"

const config = [
  {
    ignores: ["node_modules/**", ".next/**", "artifacts/**", "venv/**", "data/**", "**/__pycache__/**"],
  },
  ...next,
]

export default config
