from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class ParametrosBiorreator:
    """
    Armazena os parâmetros do modelo.
    """
    Vmax_ref: float = 5.0
    Km: float = 20.0
    Ea: float = 30000.0
    R: float = 8.314
    Tref: float = 298.15


def Vmax_Arrhenius(T, params):
    """
    Calcula Vmax(T) pela equação de Arrhenius.
    """
    T_K = T + 273.15

    Vmax = params.Vmax_ref * np.exp(
        (-params.Ea / params.R) * ((1 / T_K) - (1 / params.Tref))
    )

    return Vmax


def Biorreator_modelo(t, S, T, params):
    """
    Modelo diferencial do consumo de substrato.

    dS/dt = -Vmax(T) * S/(Km + S)
    """
    Vmax = Vmax_Arrhenius(T, params)

    dSdt = -Vmax * S[0] / (params.Km + S[0])

    return [dSdt]


def Simula_Biorreator(S0, tempo_final, T, params):
    """
    Simula o bioreator utilizando solve_ivp.
    """
    tempo = np.linspace(0, tempo_final, 300)

    solucao = solve_ivp(
        Biorreator_modelo,
        [0, tempo_final],
        [S0],
        args=(T, params),
        t_eval=tempo
    )

    concentracao = solucao.y[0]

    return tempo, concentracao
