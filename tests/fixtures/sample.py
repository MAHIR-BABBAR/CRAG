"""Module docstring for sample code fixture."""


def greet(name: str) -> str:
    return f"hello {name}"


class Greeter:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    def greet(self, name: str) -> str:
        return f"{self.prefix} {name}"
