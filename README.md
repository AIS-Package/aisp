<div align = center> 

|    <img src='https://ais-package.github.io/assets/images/logo-7b415c6841a3ed8a760eff38ecd996b8.svg'/>   |     <h1 class='text-title' align=center>**Artificial Immune Systems Package.**</h1>  |
|:-------------:|:-------------:|

</div>

---

#### Select the language / Selecione o Idioma:

<div class='language-options'>

* [English.](#english)
* [Português.](#português)

</div>

#### Package documentation / Documentação do pacote:

* [Docs.](https://ais-package.github.io/docs/intro)

* [Wiki Github.](https://github.com/AIS-Package/aisp/wiki)

---

<section id='english'>
<div align = center> 

## English

</div>

#### Summary:

> 1. [Introduction.](#introduction)
> 2. [Installation.](#installation)
>    1. [Dependencies](#dependencies)
>    2. [User installation](#user-installation)
>    3. [How to import the Techniques](#how-to-import-the-techniques)
> 3. [Examples.](#examples)

---
<section id='introduction'>

#### Introduction

The **AISP** is a python package that implements artificial immune systems techniques, distributed under the GNU Lesser General Public License v3.0 (LGPLv3).

The package started in **2022** as a research package at the Federal Institute of Northern Minas Gerais - Salinas campus (**IFNMG - Salinas**).


Artificial Immune Systems (AIS) are inspired by the vertebrate immune system, creating metaphors that apply the ability to detect and catalog pathogens, among other features of this system.

##### Algorithms implemented:

> - [x] [**Negative Selection.**](https://ais-package.github.io/docs/aisp-techniques/Negative%20Selection/)
> - [ ] *Dendritic Cells.*
> - [ ] *Clonalg.*
> - [ ] *Immune Network Theory.*

</section>

<section id='installation'>

#### **Installation**

The module requires installation of [python 3.8.10](https://www.python.org/downloads/) or higher.

<section id='dependencies'>

##### **Dependencies:**
<div align = center> 


|    Packages   |     Version   |
|:-------------:|:-------------:|
|    numpy      |    ≥ 1.22.4   |
|    scipy      |    ≥ 1.8.1    |
|    tqdm       |    ≥ 4.64.1   |

</div>

</section>
<section id='user-installation'>

##### **User installation**

The simplest way to install AISP is using ``pip``:

```Bash
pip install aisp
```

</section>
<section id='how-to-import-the-techniques'>

##### **How to import the Techniques**

``` Python
from aisp.NSA import RNSA

nsa = RNSA(N=300, r=0.05)
```

</section>
</section>
<section id='examples'>

#### Examples:

---

##### Example using the negative selection technique (**nsa**):

In the example present in this [notebook](https://github.com/AIS-Package/aisp/blob/main/examples/RNSA/example_with_randomly_generated_dataset-en.ipynb), **500** random samples were generated, arranged in two groups, one for each class.

Below are some examples that use a database for classification with the [Jupyter notebook](https://jupyter.org/) tool.


##### **Negative Selection:**

+ **RNSA** Application of negative selection techniques for classification using the Iris family flower database and Old Faithful Geyser:
    + [iris_dataBase_example](https://github.com/AIS-Package/aisp/blob/main/examples/RNSA/iris_dataBase_example_en.ipynb)
    + [geyser_dataBase_example](https://github.com/AIS-Package/aisp/blob/main/examples/RNSA/geyser_dataBase_example_en.ipynb)
+ **BNSA** 
    + [mushrooms_dataBase_example](https://github.com/AIS-Package/aisp/blob/main/examples/BNSA/mushrooms_dataBase_example_en.ipynb)

---


</section>
</section>

---

<section id='português'>
<div align = center> 

## Português

</div>

#### Sumário:

> 1. [Introdução.](#introdução)
> 2. [Instalação.](#instalação)
>    1. [Dependências](#dependências)
>    2. [Instalação do usuário](#instalação-do-usuário)
>    3. [Como importar as Tecnicas](#como-importar-as-tecnicas)
> 3. [Exemplos.](#exemplos)

---
<section id='introdução'>

#### Introdução

O **AISP** é um pacote python que implementa as técnicas dos sistemas imunológicos artificiais, distribuído sob a licença GNU Lesser General Public License v3.0 (LGPLv3).

O pacote teve início no ano de **2022** como um pacote de pesquisa no instituto federal do norte de minas gerais - campus salinas (**IFNMG - Salinas**).

Os sistemas imunológicos artificiais (SIA) inspiram-se no sistema imunológico dos vertebrados, criando metáforas que aplicam a capacidade de reconhecer e catalogar os patógenos, entre outras características desse sistema.

##### Algoritmos implementados:

> - [x] [**Seleção Negativa.**](https://ais-package.github.io/docs/aisp-techniques/Negative%20Selection/)
> - [ ] *Células Dendríticas.*
> - [ ] *Clonalg.*
> - [ ] *Teoria da Rede Imune.*

</section>

<section id='introdução'>

#### **Instalação**


O módulo requer a instalação do [python 3.8.10](https://www.python.org/downloads/) ou superior.

<section id='dependências'>

##### **Dependências:**
<div align = center> 

|    Pacotes    |     Versão    |
|:-------------:|:-------------:|
|    numpy      |    ≥ 1.22.4   |
|    scipy      |    ≥ 1.8.1    |
|    tqdm       |    ≥ 4.64.1   |

</div>
</section>

<section id='instalação-do-usuário'>

##### **Instalação do usuário**

A maneira mais simples de instalação do AISP é utilizando o ``pip``:

```Bash
pip install aisp
```

</section>

<section id='como-importar-as-tecnicas'>

##### **Como importar as Tecnicas**

``` Python
from aisp.NSA import RNSA

nsa = RNSA(N=300, r=0.05)
```

</section>
</section>
<section id='exemplos'>

#### Exemplos:

---

##### Exemplo utilizando a técnica de seleção negativa (**nsa**):

No exemplo presente nesse [notebook](https://github.com/AIS-Package/aisp/blob/main/examples/RNSA/example_with_randomly_generated_dataset-pt.ipynb), gerando **500** amostras aleatórias dispostas em dois grupos um para cada classe.

A seguir alguns exemplos que utiliza-se de base de dados para classificação com a ferramenta [Jupyter notebook](https://jupyter.org/).

#### **Seleção Negativa:**

+ **RNSA** Aplicação das tecnica de seleção negativa para classificação utilizando a base de dados de flores da família Iris e Old Faithful Geyser:
    + [iris_dataBase_example](https://github.com/AIS-Package/aisp/blob/main/examples/RNSA/iris_dataBase_example_pt-br.ipynb)
    + [geyser_dataBase_example](https://github.com/AIS-Package/aisp/blob/main/examples/RNSA/geyser_dataBase_example_pt-br.ipynb)

+ **BNSA** 
    + [mushrooms_dataBase_example](https://github.com/AIS-Package/aisp/blob/main/examples/BNSA/mushrooms_dataBase_example_en.ipynb)


---

</section>
</section>