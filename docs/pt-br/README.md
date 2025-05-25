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

O pacote foi iniciado no ano de 2022, como parte de um projeto de pesquisa desenvolvido no Instituto Federal do Norte de Minas Gerais – Campus Salinas (IFNMG – Salinas).

Os sistemas imunológicos artificiais (SIA) inspiram-se no sistema imunológico dos vertebrados, criando metáforas que aplicam a capacidade de reconhecer e catalogar os patógenos, entre outras características desse sistema.

### Algoritmos implementados:

> - [x] [**Seleção Negativa.**](https://ais-package.github.io/docs/aisp-techniques/Negative%20Selection/)
> - [x] *Algoritmos de Seleção Clonal.*
>     * [AIRS - Artificial Immune Recognition System](https://ais-package.github.io/docs/aisp-techniques/Clonal%20Selection%20Algorithms/)
> - [ ] *Células Dendríticas.*
> - [ ] *Teoria da Rede Imune.*

</section>

<section id='introdução'>

## **Instalação**


O módulo requer a instalação do [python 3.10](https://www.python.org/downloads/) ou superior.

<section id='dependências'>

### **Dependências:**
<div align = center> 

|    Pacotes    |    Versão     |
|:-------------:|:-------------:|
|     numpy     |   ≥ 1.22.4    |
|     scipy     |    ≥ 1.8.1    |
|     tqdm      |   ≥ 4.64.1    |
|     numba     |   ≥ 0.59.0    |

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

Explore os notebooks de exemplo disponíveis no repositório [AIS-Package/aisp](https://github.com/AIS-Package/aisp/tree/main/examples).
Esses notebooks demonstram como utilizar as funcionalidades do pacote em diferentes cenários, incluindo aplicações com os algoritmos
RNSA, BNSA e AIRS em conjuntos de dados como Iris, Geyser e Cogumelos.

Você pode executar os notebooks diretamente no seu navegador, sem necessidade de instalação local, utilizando o Binder:

[![Executar no Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/AIS-Package/aisp/HEAD?labpath=%2Fexamples)

> 💡 **Dica**: O Binder pode levar alguns minutos para carregar o ambiente, especialmente na primeira vez.

---

</section>