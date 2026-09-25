"""Selective Copying train using the shared Copying runner."""

from exp.copying.train import main as run
from exp.selective_copying import task


def main():
    run(task=task)


if __name__ == "__main__":
    main()
