---
id: faq
sidebar_position: 6
sidebar_label: FAQ
keywords:
    - FAQ
    - Perguntas Frequentes
    - Dúvidas
    - Ajuda
---

# Perguntas Frequentes

Soluções rápidas para possíveis dúvidas sobre o aisp.

## Uso geral

### Qual algoritmo devo escolher?

Depende do tipo de problema:

- **Detecção de anomalias**: Use `RNSA` ou `BNSA`.
  - RNSA para problemas com dados contínuos.
  - BNSA para problemas com dados binários.
- **Classificação**: Use `AIRS`, `RNSA` ou `BNSA`.
  - O `RNSA` e `BNSA`, foram implementados para serem aplicados a classificação multi-classe.
  - O `AIRS` é mais robusto para dados com ruídos.
- **Otimização**: Use `Clonalg`.
  - A implementação pode ser aplicada à otimização (min/max) de funções objetivas.
- **Clustering/Agrupamento**: Use `AiNet`.
  - Separa grupos de dados automaticamente.
  - Não requer numero de grupos predefinidos.

---

### Como normalizar meus dados para utilizar o algoritmo `RNSA`?

O RNSA trabalha exclusivamente com dados normalizados no intervalo entre (0.0, 1.0). Portanto, antes de aplicá-lo,
é necessário normalizar os dados se não estiver neste intervalo. Uma forma simples é fazer utilizando as
ferramentas de normalização do **scikit-learn**, como o
[`MinMaxScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html).

#### Exemplo

Neste exemplo, `X` representa os dados de entrada não normalizados.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

x_norm = scaler.fit_transform(X)

# Treinando o modelo com os dados normalizados.
rnsa = RNSA(N=100, r=0.1)
rnsa.fit(x_norm, y)
```

---

## Configuração de Parâmetros

### Como escolher o numero de detectores (`N`) no `RNSA` ou `BNSA`?

O número de detectores afeta diretamente a performance:

- Um número reduzido de detectores pode não cobrir adequadamente o espaço não-próprio.
- Um número muito alto de detectores pode aumentar o tempo de treinamento e pode causar overfitting.

**Recomendações**:

- Teste diferentes valores para o número de detectores até encontrar um equilíbrio adequado entre tempo de treinamento
  e desempenho do modelo.
- Utilize validação cruzada para identificar o valor que apresenta os melhores resultados de forma consistente.

---

### Qual raio (`r` ou `aff_thresh`) devo utilizar no `BNSA` ou `RNSA`?

O raio dos detectores depende da distribuição dos dados:

- Raio muito pequeno, podem não detectar anomalias.
- Raio muito grandes, podem sobrepor o self e nunca gerar detectores validos.

---

### O que é o parâmetro `r_s` no `RNSA`?

O `r_s` é o raio da amostra self. Ele define uma região ao redor de cada amostra de treinamento.

---

### Clonalg: Como definir a função objetivo?

A função objetiva deve seguir o padrão da
[classe base](https://github.com/AIS-Package/aisp/blob/main/aisp/base/core/_optimizer.py#L140).
Ela deve receber uma solução como entrada e retornar um valor do custo (ou afinidade).

```python
def affinity_function(self, solution: Any) -> float:
    pass
```

Existem duas formas de definir a função objetivo no Clonalg.

```python
def sphere(solution):
    return np.sum(solution ** 2)
```

1. Definindo a função diretamente no construtor da classe

```python
clonalg = Clonalg(
    problem_size=2,
    affinity_function=sphere
)
```

2. Utilizando o registrador de funções

```python
clonalg = Clonalg(
    problem_size=2,
)

clonalg.register("affinity_function", sphere)
```

## Informações adicionais

### Onde encontrar mais exemplos?

- [Exemplos em Jupyter Notebooks](../../examples/pt-br).

### Como contribuir para o projeto?

Veja o [Guia de Contribuição](../../CONTRIBUTING.md) no GitHub.

### Ainda tem dúvidas?

- Abra uma [**Issue no GitHub**](https://github.com/AIS-Package/aisp/issues)
