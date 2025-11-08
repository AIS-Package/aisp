# Classe base para algoritmos de otimização

## BaseOptimizer

Esta classe define a interface central para estratégias de otimização e mantém o histórico de custos, soluções
avaliadas e a melhor solução encontrada. Subclasses devem implementar os métodos `optimize` e `affinity_function`.

---

### Propriedades

#### `cost_history`

```python
@property
def cost_history(self) -> List[float]
```

Retorna o histórico de custos durante a otimização.

---

#### `solution_history`

```python
@property
def solution_history(self) -> List
```

Retorna o histórico de soluções avaliadas.

---

#### `best_solution`

```python
@property
def best_solution(self) -> Optional[Any]
```

Retorna a melhor solução encontrada até o momento, ou `None` se não disponível.

---

#### `best_cost`

```python
@property
def best_cost(self) -> Optional[float]
```

Retorna o custo da melhor solução encontrada até o momento, ou `None` se não disponível.

---

## Funções

### Função _record_best(...)

```python
def _record_best(self, cost: float, best_solution: Any) -> None
```

Registra um novo valor de custo e atualiza a melhor solução se houver melhoria.

**Parâmetros**:

* ***cost***: `float` - Valor de custo a ser adicionado ao histórico.

---

### Função get_report()

```python
def get_report(self) -> str
```

Gera um relatório resumido e formatado do processo de otimização. O relatório inclui a melhor solução, seu custo
associado e a evolução dos valores de custo por iteração.

**Retorna**:

* **report**: `str` - String formatada contendo o resumo da otimização.

---

### Função register(...)

```python
def register(self, alias: str, function: Callable[..., Any]) -> None
```

Registra dinamicamente uma função na instância do otimizador.

**Parâmetros**:

* ***alias***: `str` - Nome usado para acessar a função como atributo.
* ***function***: `Callable[..., Any]` - Callable a ser registrado.

**Exceções**:

* **TypeError**: Se `function` não for Callable.
* **AttributeError**: Se `alias` for protegido e não puder ser modificado, ou se `alias` não existir na classe do otimizador.

---

### Função reset()

```python
def reset(self)
```

Reinicia o estado interno do objeto, limpando o histórico e resetando os valores.

---

### Métodos abstratos

#### Função optimize(...)

```python
@abstractmethod
def optimize(self, max_iters: int = 50, n_iter_no_change=10, verbose: bool = True) -> Any
```

Executa o processo de otimização. Este método deve ser implementado pela subclasse para definir como a estratégia de
otimização explora o espaço de busca.

**Parâmetros**:

* ***max_iters***: `int` - Número máximo de iterações.
* ***n_iter_no_change***: `int`, padrão=10 - Número máximo de iterações sem atualização da melhor solução.
* ***verbose***: `bool`, padrão=True - Flag para habilitar ou desabilitar saída detalhada durante a otimização.

**Retorna**:

* **best_solution**: `Any` - A melhor solução encontrada pelo algoritmo de otimização.

---

#### Função affinity_function(...)

```python
def affinity_function(self, solution: Any) -> float
```

Avalia a afinidade de uma solução candidata. Este método deve ser implementado pela subclasse para definir o problema específico.

**Parâmetros**:

* ***solution***: `Any` - Solução candidata a ser avaliada.

**Retorna**:

* **cost**: `float` - Valor de custo associado à solução fornecida.
