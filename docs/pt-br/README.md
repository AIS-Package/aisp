<div align = center>

![Artificial Immune Systems Package](../assets/logos/logo.svg)

# Artificial Immune Systems Package.

Um pacote Python para algoritmos de Sistemas Imunológicos Artificiais

</div>

---

## Documentação
* [Documentação oficial](https://ais-package.github.io/pt-br/docs/intro)

---

## Introdução

O **AISP** é um pacote python que implementa as técnicas dos sistemas imunológicos artificiais, distribuído sob a
licença GNU Lesser General Public License v3.0 (LGPLv3).

O pacote foi iniciado no ano de 2022, como parte de um projeto de pesquisa desenvolvido no Instituto Federal do Norte de
Minas Gerais - Campus Salinas (IFNMG - Salinas).

Os sistemas imunológicos artificiais (SIA) inspiram-se no sistema imunológico dos vertebrados, criando metáforas que
aplicam a capacidade de reconhecer e catalogar os patógenos, entre outras características desse sistema.

---

## Algoritmos Implementados

### Seleção Negativa (`aisp.nsa`)

- **BNSA** - Algoritmo de Seleção Negativa Binária  
- **RNSA** - Algoritmo de Seleção Negativa com Valores Reais  

### Seleção Clonal (`aisp.csa`)

- **AIRS** - Sistema Imunológico Artificial de Reconhecimento  
- **CLONALG** - Algoritmo de Seleção Clonal  

### Teoria de Redes Imunológicas (`aisp.ina`)

- **AiNet** - Rede Imunológica Artificial para Agrupamento e Compressão de Dados  

### Módulo em Desenvolvimento

#### Teoria do Perigo (`aisp.dta`)

- **DCA** - Algoritmo de Células Dendríticas *(planejado)*  

## Visão geral da API

Todos os algoritmos seguem uma interface simples e consistente:

- `fit(X, y, verbose: bool = True)`: Treina o modelo para tarefas de classificação.
- `fit(X, verbose: bool = True)`: Treina o modelo para tarefas de agrupamento.
- `predict(X)`: Faz previsões com base em novos dados.
- `optimize(max_iters: int =..., n_iter_no_change: int =..., verbose: bool = True)`: executar os algoritmos de otimização

---

## Instalação

O módulo requer a instalação do [python 3.10](https://www.python.org/downloads/) ou superior.

### Dependências

<div align = center>

| Pacotes |  Versão  |
|:-------:|:--------:|
|  numpy  | ≥ 1.22.4 |
|  scipy  | ≥ 1.8.1  |
|  tqdm   | ≥ 4.64.1 |
|  numba  | ≥ 0.59.0 |

</div>

### Instalação do usuário

A maneira mais simples de instalação do AISP é utilizando o ``pip``:

```Bash
pip install aisp
```

---

## Quick Start

Abaixo estão exemplos mínimos que demonstram como usar o AISP para diferentes tarefas.

### Classificação com RNSA

```python
import numpy as np
from aisp.nsa import RNSA

# Gerando dados de treinamento
np.random.seed(1)
class_a = np.random.uniform(high=0.5, size=(50, 2))
class_b = np.random.uniform(low=0.51, size=(50, 2))
x_train = np.vstack((class_a, class_b))
y_train = ['a'] * 50 + ['b'] * 50

# Treinando o modelo
model = RNSA(N=150, r=0.3, seed=1)
model.fit(x_train, y_train, verbose=False)

# Previsão
x_test = [
    [0.15, 0.45],  # Esperado: 'a'
    [0.85, 0.65],  # Esperado: 'b'
]

y_pred = model.predict(x_test)
print(y_pred)
```

### Agrupamento com AiNet

```python
import numpy as np
from aisp.ina import AiNet

np.random.seed(1)
# Gerando dados de treinamento
a = np.random.uniform(high=0.4, size=(50, 2))
b = np.random.uniform(low=0.6, size=(50, 2))
x_train = np.vstack((a, b))

# Treinando o modelo
model = AiNet(
    N=150,
    mst_inconsistency_factor=1,
    seed=1,
    affinity_threshold=0.85,
    suppression_threshold=0.7
)

model.fit(x_train, verbose=False)

# Previsão dos rotulos de cluster
x_test = [
    [0.15, 0.45],
    [0.85, 0.65],
]

y_pred = model.predict(x_test)
print(y_pred)
```

### Otimização com CLONALG

```python
import numpy as np
from aisp.csa import Clonalg

# Definir espaço de pesquisa
bounds = {'low': -5.12, 'high': 5.12}

# Função objetiva (Rastrigin)
def rastrigin(x):
    x = np.clip(x, bounds['low'], bounds['high'])
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

# Inicialização do otimizador
model = Clonalg(problem_size=2, rate_hypermutation=0.5, bounds=bounds, seed=1)
model.register('affinity_function', rastrigin)

# Executar a otimização
population = model.optimize(100, 50, False)

print(model.best_solution, model.best_cost) # Best solution
```

---

## Exemplos

---

Explore os notebooks de exemplo disponíveis no repositório [AIS-Package/aisp](https://github.com/AIS-Package/aisp/tree/main/examples).
Esses notebooks demonstram como utilizar as funcionalidades do pacote em diferentes cenários, incluindo aplicações com os algoritmos
RNSA, BNSA e AIRS em conjuntos de dados como Iris, Geyser e Cogumelos.

Você pode executar os notebooks diretamente no seu navegador, sem necessidade de instalação local, utilizando o Binder:

[![Executar no Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/AIS-Package/aisp/HEAD?labpath=%2Fexamples)

💡 **Dica**: O Binder pode levar alguns minutos para carregar o ambiente, especialmente na primeira vez.
