"""CLI: build real homes from downloaded DLD CSV files. See app/data/dld.py for details."""
from app.data.dld import aggregate, build, load, main, prepare, to_homes  # noqa: F401

if __name__ == "__main__":
    main()
