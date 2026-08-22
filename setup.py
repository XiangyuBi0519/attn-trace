from pathlib import Path

from setuptools import find_packages, setup


def get_requirements():
    requirements_dir = Path(__file__).parent

    def _read_requirements(filename: str) -> list[str]:
        with open(requirements_dir / filename) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            elif not line.startswith("--") and not line.startswith(
                    "#") and line.strip() != "":
                resolved_requirements.append(line)
        return resolved_requirements

    try:
        requirements = _read_requirements("requirements.txt")
    except ValueError:
        print("Failed to read requirements.txt in vllm_ems.")
        requirements = []
    return requirements


setup(
    name="kv_cache_affinity",
    version="0.1.0",
    author="b30080604",
    author_email="None",
    description="vLLM ascend plugin",
    packages=find_packages(
        include=("kv_cache_affinity", "kv_cache_affinity.*"),
        exclude=("benchmarks", "examples", "tests", "wise_inference_engine", "wise_inference_engine.*"),
    ),
    python_requires='>=3.9',
    install_requires=get_requirements(),
    entry_points={
        "vllm.general_plugins": [
            "kv_cache_affinity = kv_cache_affinity.plugin:register",
        ],
    },
)
