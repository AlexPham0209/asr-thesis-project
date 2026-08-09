import pytest

from src.data.normalizer import (
    create_latex_normalizer,
    normalize_wav2vec2,
    normalize_whisper,
    contains_equation
)

@pytest.fixture(scope="module")
def wav2vec2():
    return create_latex_normalizer(normalize_wav2vec2)


@pytest.fixture(scope="module")
def whisper():
    return create_latex_normalizer(normalize_whisper)


def test_whisper_normalizer(whisper):
    original = "We minimize the above effective-Hamiltonian with respect to the parameters $r_1, r_2, r_3, r_4$, and $\theta_{12}, \theta_{13}, \theta_{14}, \theta_{23},\theta_{24}, \theta_{34}$ respectively and obtain the corresponding ground state energy $$\epsilon_\infty = -0.875837$$."
    correct_normalized = "we minimize the above effective hamiltonian with respect to the parameters $r_1, r_2, r_3, r_4$ and $\theta_{12}, \theta_{13}, \theta_{14}, \theta_{23},\theta_{24}, \theta_{34}$ respectively and obtain the corresponding ground state energy $$\epsilon_\infty = -0.875837$$"
    normalized = whisper(original)
    
    assert correct_normalized == normalized

def test_contain_equation():
    assert contains_equation("The equation $5x + 2$") == True
    assert contains_equation("The equation $$5x + 2$$") == True
    assert contains_equation("$ The way is 5") == False
