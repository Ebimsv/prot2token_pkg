import toml


def parse_requirements(file):
    """Reads requirements.txt and returns a list of dependencies."""
    with open(file) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def update_pyproject(dependencies):
    """Updates the pyproject.toml file with the dependencies."""
    # Load the existing pyproject.toml file
    pyproject = toml.load("pyproject.toml")

    # Add the dependencies from requirements.txt to the dependencies array
    pyproject["project"]["dependencies"] = dependencies

    # Write the updated pyproject.toml file
    with open("pyproject.toml", "w") as f:
        toml.dump(pyproject, f)


if __name__ == "__main__":
    # Load dependencies from requirements.txt
    dependencies = parse_requirements("requirements.txt")

    # Update pyproject.toml with the dependencies
    update_pyproject(dependencies)
