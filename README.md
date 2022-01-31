# Diagnóstico de Dados Clínicos com Machine Learning

<p align="justify">
Olá! Bem vindo ao meu repositório relativo ao projeto final do Bootcamp Data Science da Alura! Nele eu pude demonstrar todo o conhecimento adquirido durante o Bootcamp, aplicando em um caso real de dados disponibilizados pelo hospital Sírio-Libanês. O principal objetivo é a construção de um modelo de machine learning capaz de detectar quais pacientes do hospital Sírio-Libanês que foram internados, irão agravar seus casos e precisarão de leito de UTI. 

Abaixo detalherei melhor qual o escopo do projeto, quais dados serão utilizados e como será organizado o projeto.

# Descrição do Problema :clipboard:

<p align="justify">
A pandemia do COVID-19 sobrecarregou o sistema de saúde, que não estava preparado para uma demanda tão grande de leitos de UTI, profissionais capacitados, equipamentos de proteção individual, entre outros recursos. Com recursos escassos, ter em mãos a previsão de quais pacientes irão precisar utilizar esses recursos é de suma importância.

A partir do momento em que o paciente é internado no hospital Sírio-Libanês, ele(a) terá o acompanhamento dos sinais vitais e será submetido a exames de sangue. Unindo essas informações com as informações demográficas e grupo de doenças prévias do paciente, iremos prever se o paciente irá ter seu caso agravado e precisará utlizar leitos de UTI. Essa previsão deve ser o mais rápido possível, dando tempo para que os recursos da UTI possam ser organizados ou a transferência de pacientes possa ser agendada.
</p>
  
# Os dados :game_die:

<p align="justify">
Como dito anteriormente, nós temos as informações demográficas e grupo de doenças prévias do paciente, além disso também temos os sinais vitais e exames de sangue que estão sendo coletados em intervalos de 2 horas. Porém, como queremos dar a informação se o paciente precisará de um leito de UTI o mais cedo possível, iremos prever com base apenas nos sinais vitais e exames de sangue das primeiras duas horas do paciente no hospital, fazendo com que todos os profissionais do hospital fiquem cientes durante as primeiras duas horas após a internação do paciente, se aquele paciente que estão atendendo irá precisar ou não de um leito de UTI.

Os dados foram obtidos através do [Kaggle](https://www.kaggle.com/Sírio-Libanes/covid19) onde o hospital Sírio-Libanês disponibilizou os dados mencionados acima de pacientes que já foram internados, contendo tanto pacientes que foram para UTI, quanto pacientes que não agravaram seus casos e não precisaram de leitos de UTI. Afim de organizar melhor o projeto, os dados que utilizaremos foram importados e limpos neste [notebook](https://github.com/rodrigodemend/Diagnostico-de-Dados-Clinicos-com-Machine-Learning/blob/main/Notebooks/Importação_e_Limpeza_dos_Dados.ipynb) que se encontra nesse mesmo repositório. 
</p>

# Como será feito 📈

<p align="justify">
Primeiramente iremos fazer a importação dos dados do [Kaggle](https://www.kaggle.com/Sírio-Libanes/covid19) e realizar alguns procedimentos para sua limpeza. Essa etapa consiste em tratar os valores faltantes, ajustar os datatypes, verificar a presença de outliers e fazer a transformação dos dados no formato que precisamos para passar para nossos modelos de machine learning. Tudo isso se encontra nesse [notebook](https://github.com/rodrigodemend/Diagnostico-de-Dados-Clinicos-com-Machine-Learning/blob/main/Notebooks/Importação_e_Limpeza_dos_Dados.ipynb).

Após a limpeza, iremos fazer uma exploração dos dados e engenharia de atributos com o objetivo de buscar por possíveis atributos que possa ajudar nosso modelo a diferenciar entre os pacientes que irão para UTI e os que não não irão. Além de buscar por novos atributos escondidos nos dados, também iremos ajustar as escalas para que nosso modelo não de preferência para um atributo apenas porque está em uma escala diferente dos demais. Está e as demais etapas se encontram nesse NOTEBOOK.
  
Também iremos fazer o balanceamento das classes utilizando uma técnica conhecida como SMOTE. O balanceamento dos dados é algo importante pois se tivermos muito mais dados de apenas uma classe, nosso modelo poderá prever tudo para essa classe e mesmo assim ele ainda terá uma suposta boa avaliação de performace.

Uma vez definidos os atributos que iremos treinar nosso modelo, vamos passar para uma etapa muito importante que é a seleção dos melhores atributos. Aqui, iremos reduzir significativamente a quantidade de atributos que nosso modelo irá utilizar para treinamento, aumentando a capacidade de generalização do modelo e se tornando mais fácil para implementação em produção.

Chegou enfim o momento da criação do modelo de machine learning, porém antes de partir para modelo mais complexos e otimizações, iremos criar uma Regressão Logística básica como baseline. Esse modelo é importante para conseguirmos comparar se as melhorias que estamos fazendo nos nosso modelos estão surtindo efeito ou não.

Após a criação da baseline, iremos criar um modelo baseado na própria Regressão Logística, mas dessa vez iremos fazer uma seleção mais elaborada dos atributos, eliminando os atributos correlacionados e aplicando a técnica do Step Backward Selection para selecionar apenas os atributos que são importantes para a Regressão Logística. Além disso iremos também trabalhar na otimização dos hiperparâmetros e vamos fazer uma análise nos dados de validação que nosso modelo errou, buscando por padrões que poderemos criar novos atributos que ajudarão nosso modelo a melhorar a performace.
  
Agora, iremos seguir os mesmos passos da criação do modelo da Regressão Logística para criar uma Floresta Aleatória. Após a criação e otimização dos dois modelos, iremos fazer uma comparação entre eles a fim de escolher qual melhor se adequa para solucionar nosso problema.
</p>

======================================================================================================================

Na pasta notebooks desse repositório, existe um [notebook](https://github.com/rodrigodemend/Diagnostico-de-Dados-Clinicos-com-Machine-Learning/blob/main/Notebooks/Importação_e_Limpeza_dos_Dados.ipynb) sobre a importação e limpeza dos dados, assim como um notebook sobre a modelagem do problema e a criação do modelo de machine learning. Além disso, também existe um notebook referente as funções criadas durante o projeto a fim de melhorar a organização do projeto e reutilização de códigos.

======================================================================================================================





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
