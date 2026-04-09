import app


def main() -> None:
    config = app.get_config()
    cache_dir, _, _ = app.ensure_runtime_dirs(config)
    model = app.load_model(config["model_repo"], str(cache_dir))
    print("SELF-CHECK OK: model loaded")
    print("model_repo:", config["model_repo"])
    print("cache_dir:", cache_dir)
    print("model_type:", type(model).__name__)


if __name__ == "__main__":
    main()
