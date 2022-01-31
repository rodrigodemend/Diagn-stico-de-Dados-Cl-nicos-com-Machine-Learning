# Diagnóstico de Dados Clinicos com Machine Learning

Olá! Bem vindo ao meu repositório relativo ao projeto final do Bootcamp Data Science da Alura! Nele eu pude demonstrar todo o conhecimento adquirido durante o Bootcamp, aplicando em um caso real de dados disponibilizados pelo hospital Sírio-Libanês. O principal objetivo é a construção de um modelo de machine learning capaz de detectar quais pacientes do hospital Sírio-Libanês que foram internados, irão agravar seus casos e precisarão de leito de UTI. 

Abaixo detalherei melhor qual o escopo do projeto, quais dados serão utilizados e como será organizado o projeto.

# Descrição do Problema :clipboard:

A pandemia do COVID-19 sobrecarregou o sistema de saúde, que não estava preparado para uma demanda tão grande de leitos de UTI, profissionais capacitados, equipamentos de proteção individual, entre outros recursos. Com recursos escassos, ter em mãos a previsão de quais pacientes irão precisar utilizar esses recursos é de suma importância.

A partir do momento em que o paciente é internado no hospital Sírio-Libanês, ele(a) terá o acompanhamento dos sinais vitais e será submetido a exames de sangue. Unindo essas informações com as informações demográficas e grupo de doenças prévias do paciente, iremos prever se o paciente irá ter seu caso agravado e precisará utlizar leitos de UTI. Essa previsão deve ser o mais rápido possível, dando tempo para que os recursos da UTI possam ser organizados ou a transferência de pacientes possa ser agendada.

# Os dados :game_die:

Como dito anteriormente, nós temos as informações demográficas e grupo de doenças prévias do paciente, além disso também temos os sinais vitais e exames de sangue que estão sendo coletados em intervalos de 2 horas. Porém, como queremos dar a informação se o paciente precisará de um leito de UTI o mais cedo possível, iremos prever com base apenas nos sinais vitais e exames de sangue das primeiras duas horas do paciente no hospital, fazendo com que todos os profissionais do hospital fiquem cientes durante as primeiras duas horas após a internação do paciente, se aquele paciente que estão atendendo irá precisar ou não de um leito de UTI.

Os dados foram obtidos através do [Kaggle](https://www.kaggle.com/Sírio-Libanes/covid19) onde o hospital Sírio-Libanês disponibilizou os dados mencionados acima de pacientes que já foram internados, contendo tanto pacientes que foram para UTI, quanto pacientes que não agravaram seus casos e não precisaram de leitos de UTI. 

Afim de organizar melhor o projeto, os dados que utilizaremos foram importados e limpos neste [Notebook](https://github.com/rodrigodemend/Diagnostico-de-Dados-Clinicos-com-Machine-Learning/blob/main/Notebooks/Importação_e_Limpeza_dos_Dados.ipynb) que se encontra nesse mesmo repositório. 











======================================================================================================================

Na pasta notebooks desse repositório, existe um [notebook](https://github.com/rodrigodemend/Diagnostico-de-Dados-Clinicos-com-Machine-Learning/blob/main/Notebooks/Importação_e_Limpeza_dos_Dados.ipynb) sobre a importação e limpeza dos dados, assim como um notebook sobre a modelagem do problema e a criação do modelo de machine learning. Além disso, também existe um notebook referente as funções criadas durante o projeto a fim de melhorar a organização do projeto e reutilização de códigos.

======================================================================================================================





# Os dados :game_die:


O conjunto de dados diz a respeito do número de novas mortes por Covid-19 de 25 de Fevereiro de 2020 até 20 de Dezembro de 2021.

Os dados foram obtidos através do [Brasil.IO](https://brasil.io/dataset/covid19/caso_full/) onde estão sendo disponibilizados boletins informativos sobre os casos do coronavírus. Afim de organizar melhor o projeto, os dados que utilizaremos foram importados e limpos neste [Notebook](https://github.com/rodrigodemend/Previsao_Covid/blob/main/Notebooks/Importação_e_Limpeza_dos_dados_de_Covid_19.ipynb/) que se encontra nesse mesmo repositório. 

# Como será feito 📈

Primeiramente iremos criar um modelo básico sem otimização alguma com o Prophet. Após isso vamos criar diversos experimentos onde vamos otimizar os principais parâmetros da tendência, sazonalidade, feriados e outliers do modelo. Chegando assim em um modelo com erro bem menor do que o modelo inicial.

# Resultados :dart:

Conseguimos otimizar o modelo e diminuir em mais de 3x as duas métricas que usamos de avaliação do modelo (MAE e RMSE). Com isso chegamos com um erro menor de 6 mortes por dia de previsão, portanto temos uma previsão confiável do número de mortes por Covid-19 em Santa Catarina nos próximos dias.

# Conclusões :memo:

Através de melhores estudos e aperfeiçoamento dos parâmetros, é possível encontrar modelos muitos precisos. Neste caso conseguimos otimizar bem, porém acredito que ainda exista um grande campo de otimização desse modelo para que consiga prever as mortes por Covid-19 com cada vez mais eficiência.

Link para o [Notebook](https://github.com/rodrigodemend/Previsao_Covid/blob/main/Notebooks/Previsão_de_Series_Temporais_usando_Prophet.ipynb) do projeto.

## Autor 🧔

Rodrigo de Mendonça

Linkedin: https://www.linkedin.com/in/rodrigomendonça/
