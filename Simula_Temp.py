from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class params:
  x0: float = 0.5
  xe: float = 0.05

  @staticmethod
  def constante_cinetica(T):

      k = 0.2 * (1 + 0.02 * (T - 60))

      return k


def modelo_secagem(t, X, params, T):

    k = params.constante_cinetica(T)

    dXdt = -k * (X[0] - params.xe)

    return [dXdt]


def simular_secagem(T, params):

    # intervalo de tempo
    t_inicial = 0
    t_final = 20

    # pontos para avaliação
    t_eval = np.linspace(t_inicial, t_final, 200)

    # resolução da EDO
    solucao = solve_ivp(
        modelo_secagem,
        [t_inicial, t_final],
        [params.x0],
        t_eval=t_eval,
        args=(params, T)
    )

    return solucao
