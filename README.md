<div align = center> 

|    <img src='https://ais-package.github.io/assets/images/logo-7b415c6841a3ed8a760eff38ecd996b8.svg'/>   |     <h1 class='text-title' align=center>**Artificial Immune Systems Package.**</h1>  |
|:-------------:|:-------------:|

</div>

---

#### Select the language / Selecione o Idioma:

<div class='language-options'>

* [English.](#english)
* [Português.](https://ais-package.github.io/pt-br/docs/intro)

</div>

#### Package documentation / Documentação do pacote:

* [Docs.](https://ais-package.github.io/docs/intro)

* [Wiki Github.](https://github.com/AIS-Package/aisp/wiki)

---

<section id='english'>

#### Summary:

> 1. [Introduction.](#introduction)
> 2. [Installation.](#installation)
>    1. [Dependencies](#dependencies)
>    2. [User installation](#user-installation)
> 3. [Examples.](#examples)

---
<section id='introduction'>

#### Introduction

The **AISP** is a python package that implements artificial immune systems techniques, distributed under the GNU Lesser General Public License v3.0 (LGPLv3).

The package started in **2022** as a research package at the Federal Institute of Northern Minas Gerais - Salinas campus (**IFNMG - Salinas**).


Artificial Immune Systems (AIS) are inspired by the vertebrate immune system, creating metaphors that apply the ability to detect and catalog pathogens, among other features of this system.

##### Algorithms implemented:

> - [x] [**Negative Selection.**](https://ais-package.github.io/docs/aisp-techniques/Negative%20Selection/)
> - [ ] *Clonal Selection Algorithms.*
> - [ ] *Dendritic Cells.*
> - [ ] *Immune Network Theory.*

</section>

<section id='installation'>

#### **Installation**

The module requires installation of [python 3.10](https://www.python.org/downloads/) or higher.

<section id='dependencies'>

##### **Dependencies:**
<div align = center> 


|    Packages   |     Version   |
|:-------------:|:-------------:|
|    numpy      |    ≥ 1.22.4   |
|    scipy      |    ≥ 1.8.1    |
|    tqdm       |    ≥ 4.64.1   |
|    numba      |    ≥ 0.59.0   |

</div>

</section>
<section id='user-installation'>

##### **User installation**

The simplest way to install AISP is using ``pip``:

```Bash
pip install aisp
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