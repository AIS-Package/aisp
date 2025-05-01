## Sumário:

>1. [Introdução.](#introdução)
>2. [Instalação.](#instalação)
>    1. [Dependências](#dependências)
>    2. [Instalação do usuário](#instalação-do-usuário)
>3. [Exemplos.](#exemplos)

---

<section id='introdução'>

## Introdução

O **AISP** é um pacote python que implementa as técnicas dos sistemas imunológicos artificiais, distribuído sob a licença GNU Lesser General Public License v3.0 (LGPLv3).

O pacote teve início no ano de **2022** como um pacote de pesquisa no instituto federal do norte de minas gerais - campus salinas (**IFNMG - Salinas**).

Os sistemas imunológicos artificiais (SIA) inspiram-se no sistema imunológico dos vertebrados, criando metáforas que aplicam a capacidade de reconhecer e catalogar os patógenos, entre outras características desse sistema.

### Algoritmos implementados:

> - [x] [**Seleção Negativa.**](https://ais-package.github.io/docs/aisp-techniques/Negative%20Selection/)
> - [ ] *Algoritmos de Seleção Clonal.*
> - [ ] *Células Dendríticas.*
> - [ ] *Teoria da Rede Imune.*

</section>

<section id='introdução'>

## **Instalação**


O módulo requer a instalação do [python 3.10](https://www.python.org/downloads/) ou superior.

<section id='dependências'>

### **Dependências:**
<div align = center> 

|    Pacotes    |     Versão    |
|:-------------:|:-------------:|
|    numpy      |    ≥ 1.22.4   |
|    scipy      |    ≥ 1.8.1    |
|    tqdm       |    ≥ 4.64.1   |
|    numba      |    ≥ 0.59.0   |

</div>
</section>

<section id='instalação-do-usuário'>

### **Instalação do usuário**

A maneira mais simples de instalação do AISP é utilizando o ``pip``:

```Bash
pip install aisp
```

</section>

</section>
<section id='exemplos'>

## Exemplos:

---

### Exemplo utilizando a técnica de seleção negativa (**nsa**):

No exemplo presente nesse [notebook](https://github.com/AIS-Package/aisp/blob/main/examples/RNSA/example_with_randomly_generated_dataset-pt.ipynb), gerando **500** amostras aleatórias dispostas em dois grupos um para cada classe.

A seguir alguns exemplos que utiliza-se de base de dados para classificação com a ferramenta [Jupyter notebook](https://jupyter.org/).

## **Seleção Negativa:**

+ **RNSA** Aplicação das tecnica de seleção negativa para classificação utilizando a base de dados de flores da família Iris e Old Faithful Geyser:
    + [iris_dataBase_example](https://github.com/AIS-Package/aisp/blob/main/examples/RNSA/iris_dataBase_example_pt-br.ipynb)
    + [geyser_dataBase_example](https://github.com/AIS-Package/aisp/blob/main/examples/RNSA/geyser_dataBase_example_pt-br.ipynb)

+ **BNSA** 
    + [mushrooms_dataBase_example](https://github.com/AIS-Package/aisp/blob/main/examples/BNSA/mushrooms_dataBase_example_en.ipynb)


---

</section>