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
> - [x] [**Clonal Selection Algorithms.**](https://ais-package.github.io/docs/aisp-techniques/Clonal%20Selection%20Algorithms/)
>     * [AIRS - Artificial Immune Recognition System](https://ais-package.github.io/docs/aisp-techniques/Clonal%20Selection%20Algorithms/airs/)
> - [ ] *Danger Theory.*
> - [x] [*Immune Network Theory.*](https://ais-package.github.io/docs/aisp-techniques/Immune%20Network%20Theory)
>   - [AiNet - Artificial Immune Network para Clustering and Compression](https://ais-package.github.io/docs/aisp-techniques/Immune%20Network%20Theory/ainet)

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

Explore the example notebooks available in the [AIS-Package/aisp repository](https://github.com/AIS-Package/aisp/tree/main/examples).
These notebooks demonstrate how to utilize the package's functionalities in various scenarios, including applications of the RNSA,
BNSA and AIRS algorithms on datasets such as Iris, Geyser, and Mushrooms.

You can run the notebooks directly in your browser without any local installation using Binder:

[![Launch on Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/AIS-Package/aisp/HEAD?labpath=%2Fexamples)

> 💡 **Tip**: Binder may take a few minutes to load the environment, especially on the first launch.
</section>
</section>