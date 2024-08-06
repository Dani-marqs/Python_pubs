import pandas as pd
import numpy as np
import seaborn as srn
import statsmodels.formula.api as sm
import statsmodels.stats.api as sms
import statsmodels.api as sma
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import boxcox
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor


data = pd.read_csv(r"C:\Users\Dell\Desktop\Danilo\UFOP\TCC\data\bq-results-20221004-122042-1664886083040.csv")
dic = pd.read_csv(r"C:\Users\Dell\Desktop\Danilo\UFOP\TCC\data\dictionary.csv")
pop = pd.read_csv(r"C:\Users\Dell\Desktop\Danilo\UFOP\TCC\data\populacao_MG.csv")

srn.set(rc={'figure.figsize' : (9, 5)})

data.rename(columns = {'quantidade_horas_contratadas' : 'horas_cont'}, inplace = True)
data.drop(['ano'], axis = 1, inplace = True)


pop.drop(['RISP', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2020', '2021'], axis = 1, inplace = True)

pop.sort_values(by = '2019', ascending = False, inplace = True)
pop_remove = pop.loc[pop['2019'] < 100000]
pop = pop.drop(pop_remove.index)

data['salary_log'] = np.log(data['salario_mensal'])
data['idade_log'] = np.log(data['idade'])
data['idade2_log'] = np.log(data['idade_2'])

data['idade_2'] = data['idade'] ** 2
data['salary_q'] = data['salario_mensal'] ** 0.5

#configurar qualitativas
data.loc[data['grau_instrucao'] == 1, 'grau_instrucao'] = '*Analfabeto'
data.loc[data['grau_instrucao'] == 2, 'grau_instrucao'] = 'Até 5ª incompleto'
data.loc[data['grau_instrucao'] == 3, 'grau_instrucao'] = '5ª completo'
data.loc[data['grau_instrucao'] == 4, 'grau_instrucao'] = '6ª a 9ª Fundamental'
data.loc[data['grau_instrucao'] == 5, 'grau_instrucao'] = 'Fundamental Completo'
data.loc[data['grau_instrucao'] == 6, 'grau_instrucao'] = 'Fundamental Completo'
data.loc[data['grau_instrucao'] == 7, 'grau_instrucao'] = 'Médio Completo'
data.loc[data['grau_instrucao'] == 8, 'grau_instrucao'] = 'Superior Incompleto'
data.loc[data['grau_instrucao'] == 9, 'grau_instrucao'] = 'Superior Completo'
data.loc[data['grau_instrucao'] == 10, 'grau_instrucao'] = 'Mestrado'
data.loc[data['grau_instrucao'] == 11, 'grau_instrucao'] = 'Doutorado'

data.loc[data['grau_instrucao'] == '*Analfabeto', 'grau_instrucao'] = 'Analf.'
data.loc[data['grau_instrucao'] == 'Fundamental Incompleto', 'grau_instrucao'] = 'Fundam. inc.'
data.loc[data['grau_instrucao'] == 'Fundamental Completo', 'grau_instrucao'] = 'Fundam. Comp.'
data.loc[data['grau_instrucao'] == 'Médio Completo', 'grau_instrucao'] = 'Médio'
data.loc[data['grau_instrucao'] == 'Superior Incompleto', 'grau_instrucao'] = 'Sup. Inc.'
data.loc[data['grau_instrucao'] == 'Superior Completo', 'grau_instrucao'] = 'Superior'


#qualitativa sexo
data.loc[data['sexo'] == 1, 'sexo'] = 'Masculino'
data.loc[data['sexo'] == 2, 'sexo'] = 'Feminino'

#qualitativa raca_cor
data.loc[data['raca_cor'] == 1, 'raca_cor'] = 'Indigena'
data.loc[data['raca_cor'] == 2, 'raca_cor'] = '*Branca'
data.loc[data['raca_cor'] == 4, 'raca_cor'] = 'Preta'
data.loc[data['raca_cor'] == 6, 'raca_cor'] = 'Amarela'
data.loc[data['raca_cor'] == 8, 'raca_cor'] = 'Parda'
data.loc[data['raca_cor'] == 9, 'raca_cor'] = 'Nao Ident'
data.loc[data['raca_cor'] == -1, 'raca_cor'] = 'Ignorado'

data.loc[data['raca_cor'] == 'Parda', 'raca_cor'] = 'Pardos'
data.loc[data['raca_cor'] == '*Branca', 'raca_cor'] = '*Brancos'
data.loc[data['raca_cor'] == 'Preta', 'raca_cor'] = 'Negros'
data.loc[data['raca_cor'] == 'Amarela', 'raca_cor'] = 'Amarelos'

#Removendo valores não significativos
data.loc[data['raca_cor'] == 'Nao Ident']
data.loc[data['raca_cor'] == 'Indigena']
df_remove = data.loc[data['raca_cor'].isin(['Nao Ident', 'Indigena'])]
data = data.drop(df_remove.index)


#Aplicando ID_MUNICIPIO
print(pop['MUNICÍPIO'])
pop['ID_MUNICIPIO'] = 1
pop.loc[pop['MUNICÍPIO'] == 'BELO HORIZONTE', 'ID_MUNICIPIO'] = 3106200
pop.loc[pop['MUNICÍPIO'] == 'UBERLANDIA', 'ID_MUNICIPIO'] = 3170206
pop.loc[pop['MUNICÍPIO'] == 'CONTAGEM', 'ID_MUNICIPIO'] = 3118601
pop.loc[pop['MUNICÍPIO'] == 'JUIZ DE FORA', 'ID_MUNICIPIO'] = 3136702
pop.loc[pop['MUNICÍPIO'] == 'BETIM', 'ID_MUNICIPIO'] = 3106705
pop.loc[pop['MUNICÍPIO'] == 'MONTES CLAROS', 'ID_MUNICIPIO'] = 3143302
pop.loc[pop['MUNICÍPIO'] == 'RIBEIRAO DAS NEVES', 'ID_MUNICIPIO'] = 3154606
pop.loc[pop['MUNICÍPIO'] == 'UBERABA', 'ID_MUNICIPIO'] = 3170107
pop.loc[pop['MUNICÍPIO'] == 'GOVERNADOR VALADARES', 'ID_MUNICIPIO'] = 3127701
pop.loc[pop['MUNICÍPIO'] == 'IPATINGA', 'ID_MUNICIPIO'] = 3131307
pop.loc[pop['MUNICÍPIO'] == 'SETE LAGOAS', 'ID_MUNICIPIO'] = 3167202
pop.loc[pop['MUNICÍPIO'] == 'DIVINOPOLIS', 'ID_MUNICIPIO'] = 3122306
pop.loc[pop['MUNICÍPIO'] == 'SANTA LUZIA', 'ID_MUNICIPIO'] = 3157807
pop.loc[pop['MUNICÍPIO'] == 'IBIRITE', 'ID_MUNICIPIO'] = 3129806
pop.loc[pop['MUNICÍPIO'] == 'POCOS DE CALDAS', 'ID_MUNICIPIO'] = 3151800
pop.loc[pop['MUNICÍPIO'] == 'PATOS DE MINAS', 'ID_MUNICIPIO'] = 3148004
pop.loc[pop['MUNICÍPIO'] == 'POUSO ALEGRE', 'ID_MUNICIPIO'] = 3152501
pop.loc[pop['MUNICÍPIO'] == 'TEOFILO OTONI', 'ID_MUNICIPIO'] = 3168606
pop.loc[pop['MUNICÍPIO'] == 'BARBACENA', 'ID_MUNICIPIO'] = 3105608
pop.loc[pop['MUNICÍPIO'] == 'SABARA', 'ID_MUNICIPIO'] = 3156700
pop.loc[pop['MUNICÍPIO'] == 'VARGINHA', 'ID_MUNICIPIO'] = 3170701
pop.loc[pop['MUNICÍPIO'] == 'CONSELHEIRO LAFAIETE', 'ID_MUNICIPIO'] = 3118304
pop.loc[pop['MUNICÍPIO'] == 'VESPASIANO', 'ID_MUNICIPIO'] = 3171204
pop.loc[pop['MUNICÍPIO'] == 'ITABIRA', 'ID_MUNICIPIO'] = 3131703
pop.loc[pop['MUNICÍPIO'] == 'ARAGUARI', 'ID_MUNICIPIO'] = 3103504
pop.loc[pop['MUNICÍPIO'] == 'UBA', 'ID_MUNICIPIO'] = 3169901
pop.loc[pop['MUNICÍPIO'] == 'PASSOS', 'ID_MUNICIPIO'] = 3147907
pop.loc[pop['MUNICÍPIO'] == 'CORONEL FABRICIANO', 'ID_MUNICIPIO'] = 3119401
pop.loc[pop['MUNICÍPIO'] == 'MURIAE', 'ID_MUNICIPIO'] = 3143906
pop.loc[pop['MUNICÍPIO'] == 'ARAXA', 'ID_MUNICIPIO'] = 3104007
pop.loc[pop['MUNICÍPIO'] == 'ITUIUTABA', 'ID_MUNICIPIO'] = 3134202
pop.loc[pop['MUNICÍPIO'] == 'LAVRAS', 'ID_MUNICIPIO'] = 3138203
pop.loc[pop['MUNICÍPIO'] == 'NOVA SERRANA', 'ID_MUNICIPIO'] = 3145208
 
#Extraindo ID das cidades grandes

extracted_id = pop['ID_MUNICIPIO']

#Apicando booleana em CENTRO_URB

data['centro_urb'] = 0
data.loc[data['id_municipio'].isin(extracted_id), 'centro_urb'] = 1
data.loc[data['centro_urb'] == 0]
data.loc[data['centro_urb'] == 1]

#barplot variáveis
horas_agrup = data.groupby(['horas_cont']).size()
horas_agrup.plot.bar(color = 'gray')

cidade_agrup = data.groupby(['centro_urb']).size()
cidade_agrup.plot.bar(color = 'gray')

sexo_agrup = data.groupby(['sexo']).size()
sexo_agrup.plot.bar(color = 'gray')

raca_agrup = data.groupby(['raca_cor']).size()
raca_agrup.plot.bar(color = 'gray')

esc_agrup = data.groupby(['grau_instrucao']).size()
esc_agrup.plot.bar(color = 'gray') 

idade_agrup = data.groupby(['idade']).size()
idade_agrup.plot.bar(color = 'gray')

df_remove2 = data.loc[data['salario_mensal'] == 0]
df_remove3 = data.loc[data['salario_mensal'] > 5000]
df_remove4 = data.loc[data['horas_cont'] != 44]

df_removeX = data.loc[(data['salario_mensal'] > 1100) &
                      (data['salario_mensal'] < 1050)]
#Apenas 2901 OBS idade acima de 70
#df_remove4 = data.loc[data['idade'] > 70]
#APENAS 43K OBSERVAÇÕES ACIMA DE 5K SALÁRIO EM 2.8M DE OBS
print((data.loc[data['salario_mensal'] > 1800]))
print((data.loc[data['salario_mensal'] < 800]))
df_remove4 = data.loc[data['horas_cont'] != 44]
data = data.drop(df_remove2.index)
data = data.drop(df_remove3.index)
data = data.drop(df_remove4.index)
df_remove5 = data.loc[data['idade'] < 18]
data = data.drop(df_remove5.index)


data['idade'] = data['idade'].astype(float).astype(int)
srn.distplot(data['idade'], kde=True)
print(data['idade'].describe())
print(data['salario_mensal'].describe())

srn.boxplot(data['salario_mensal'])
srn.distplot(data['salario_mensal'])

srn.distplot(data['salario_mensal'])
srn.distplot(data['salary_log'])

srn.distplot(data['salary_q'])


plt.hist(data['salario_mensal'], color = 'gray')
print(data.loc[data['salario_mensal'] > 10000])

srn.boxplot(data['horas_cont'])

hour_1 = data.loc[data['horas_cont'] == 1]
horas = [1, 20, 22, 24, 30, 36, 40, 42, 44]
data = data.loc[data['horas_cont'].isin(horas)]
data.isnull().sum()    

srn.distplot(data['idade'], kde=True)
plt.hist(data['idade'], color = 'gray')


#SECOND SESSION
#análise de agrupamento raca_cor
#data.loc[data['raca_cor'] == 'Amarela', 'raca_cor'] = 'Indigena/Amarela'
#data.loc[data['raca_cor'] == 'Indigena', 'raca_cor'] = 'Indigena/Amarela'

srn.boxplot(x = data['raca_cor'], y = data['salario_mensal'], color = 'Skyblue')

#agrupamento grau_instrucao
data.loc[data['grau_instrucao'].isin(['5ª completo', '6ª a 9ª Fundamental', 'Até 5ª incompleto']), 'grau_instrucao'] = 'Fundamental Incompleto'

#--------------------- // ---------------------- // ---------------------

data.to_csv(r"C:\Users\Dell\Desktop\Danilo\UFOP\TCC\data\Base_smout.csv", index=False)
data = pd.read_csv(r"C:\Users\Dell\Desktop\Danilo\UFOP\TCC\data\Base_bruta.csv")
#--------------------- // ---------------------- // ---------------------

data = pd.read_csv(r"C:\Users\Dell\Desktop\Danilo\UFOP\TCC\data\base_smout.csv")



#ESTATÍSTICA DESCRITIVA
srn.boxplot(data['salario_mensal'], color = 'Skyblue')
data['salario_mensal'].describe()
data['idade'].describe()

srn.set(style = 'darkgrid')

branco = data.loc[data['raca_cor'] == 'Brancos']
srn.boxplot(branco['salario_mensal'], color = 'Gray')
branco['salario_mensal'].describe()

pardo = data.loc[data['raca_cor'] == 'Pardos']
srn.boxplot(pardo['salario_mensal'], color = 'Gray')
pardo['salario_mensal'].describe()

negro = data.loc[data['raca_cor'] == 'Negros']
srn.boxplot(negro['salario_mensal'], color = 'Gray')
negro['salario_mensal'].describe()

amarelo = data.loc[data['raca_cor'] == 'Amarelos']
srn.boxplot(amarelo['salario_mensal'], color = 'Gray')
amarelo['salario_mensal'].describe()

#escolaridade
analf = data.loc[data['grau_instrucao'] == 'Analf.']
analf['salario_mensal'].describe()

fundin = data.loc[data['grau_instrucao'] == 'Fundam. inc.']
fundin['salario_mensal'].describe()

fund = data.loc[data['grau_instrucao'] == 'Fundam. Comp.']
fund['salario_mensal'].describe()

med = data.loc[data['grau_instrucao'] == 'Médio']
med['salario_mensal'].describe()

supin = data.loc[data['grau_instrucao'] == 'Sup. Inc.']
supin['salario_mensal'].describe()

sup = data.loc[data['grau_instrucao'] == 'Superior']
sup['salario_mensal'].describe()

#gênero
masc = data.loc[data['sexo'] == 'Masculino']
masc['salario_mensal'].describe()

fem = data.loc[data['sexo'] == 'Feminino']
fem['salario_mensal'].describe()

dd = branco.loc[branco['grau_instrucao'] == '*Analfabeto']
len(dd)


srn.boxplot(x = data['raca_cor'], y = data['salario_mensal'], color = 'Skyblue')
srn.boxplot(x = data['grau_instrucao'], y = data['salario_mensal'], color = 'Skyblue')
srn.boxplot(x = data['sexo'], y = data['salario_mensal'], color = 'Skyblue')

srn.barplot(x = data['grau_instrucao'], y = data['raca_cor'], color = 'Skyblue')

## Escolaridade por Gênero
esc_masc = masc.groupby(['grau_instrucao']).size()
emp = esc_masc/len(masc)
plt.subplot(1,2,1)
emp.plot.bar(title = 'Masculino', color = 'Skyblue', xlabel = " ")
#esc_masc.plot.bar(title = 'Masculino', color = 'Skyblue')

esc_fem = fem.groupby(['grau_instrucao']).size()
efp = esc_fem/len(fem)
plt.subplot(1,2,2)
efp.plot.bar(title = 'Feminino', color = 'Skyblue', xlabel = " ")

## Escolaridade por Cor
esc_branc = branco.groupby(['grau_instrucao']).size()
plt.subplot(2,2,1)
plt.tick_params(labelbottom = False)
propb = esc_branc/len(branco) 
propb.plot.bar(title = 'Brancos', color = 'Skyblue', xlabel = " ")
#esc_branc.plot.bar(title = 'Brancos', color = 'Skyblue', xlabel = " ")

esc_negro = negro.groupby(['grau_instrucao']).size()
plt.subplot(2,2,2)
plt.tick_params(labelbottom = False)
propn = esc_negro/len(negro) 
propn.plot.bar(title = 'Negros', color = 'Skyblue', xlabel = " ")
#esc_negro.plot.bar(title = 'Negros', color = 'Skyblue', xlabel = " ")

esc_pardo = pardo.groupby(['grau_instrucao']).size()
plt.subplot(2,2,3)
propp = esc_pardo/len(pardo) 
propp.plot.bar(title = 'Pardos', color = 'Skyblue')
#esc_pardo.plot.bar(title = 'Pardos', color = 'Skyblue')

esc_amar = amarelo.groupby(['grau_instrucao']).size()
plt.subplot(2,2,4)
propa = esc_amar/len(amarelo) 
propa.plot.bar(title = 'Amarelos', color = 'Skyblue')
#esc_amar.plot.bar(title = 'Amarelos', color = 'Skyblue')




srn.distplot(data['idade'])
srn.distplot(data['salario_mensal'])

#NORMALTEST STATSMODEL
print(normaltest(data['salary_log']))
print(normaltest(data['salario_mensal']))

data['bcox'] = boxcox(data['salario_mensal'])[0]
srn.distplot(data['bcox'])
print(normaltest(data['bcox']))

#SAMPLE
sample = data.sample(frac =0.01)
s_remove = sample.loc[sample['salario_mensal'] < 998]
sample = sample.drop(s_remove.index)
srn.distplot(sample['salario_mensal'])

print(normaltest(sample['salario_mensal']))


#REMOVENDO OUTLIERS
Q1 = np.percentile(data['salario_mensal'], 25, interpolation = 'midpoint') 
Q3 = np.percentile(data['salario_mensal'], 75, interpolation = 'midpoint')
IQR = Q3 - Q1
# Upper bound
upper = np.where(data['salario_mensal'] >= (Q3+1.5*IQR))
upremov = data.loc[data['salario_mensal'] >= (Q3+1.5*IQR)]
# Lower bound
lower = np.where(data['salario_mensal'] <= (Q1-1.5*IQR))
loremov = data.loc[data['salario_mensal'] <= (Q1-1.5*IQR)]
 
#''' Removing the Outliers '''
data.drop(upper[0], inplace = True)
data.drop(lower[0], inplace = True)

data = data.drop(upremov.index)
data = data.drop(loremov.index)
srn.distplot(data['salario_mensal'])

#ROBUST regression model
rmodel = sma.RLM(data['salario_mensal'], data[['idade', 'idade_2', 'centro_urb']])
rmodel_t = rmodel.fit()
rmodel_t.summary()

#OLS model
model = sm.ols(formula = 'salary_log ~ grau_instrucao + idade + idade_2 + sexo + raca_cor + centro_urb', data = data)
model_t = model.fit()
model_t = model.fit(cov_type = "HC0")
model_t.summary()
    
#data frame vazio para mestrado e doutorado
data.loc[data['grau_instrucao'] == 'Doutorado']
data.loc[data['grau_instrucao'] == 'Mestrado']

#Teste Homecedasticidade
print('GOLDFELD-QUANDT TEST:', round(sms.het_goldfeldquandt(model_t.resid, model_t.model.exog)[0], 3))

print('P VALEUe GQ TEST:', sms.het_goldfeldquandt(model_t.resid, model_t.model.exog)[1])

#Teste Normalidade De Distribuição (S_WILK)
print(shapiro(data['salario_mensal']))
print(shapiro(data['salary_log']))
print(shapiro(data['idade']))

#TESTEs VIF e SHAPIRO - NESCESSÁRIO ACRESCENTAR ESCOLARIDADE NUMÉRICA
X = data[['grau_instrucao', 'sexo', 'idade', 'centro_urb', 'raca_cor']]
X['sexo'] = X['sexo'].map({'Masculino' : 1, 'Feminino': 0})
X['raca_cor'] = X['raca_cor'].map({'*Branca' : 0, 'Amarela' : 1, 'Preta' : 2,
                                   'Parda' : 3})
X['grau_instrucao'] = X['grau_instrucao'].map({'*Analfabeto' : 0, 
                                               'Fundamental Incompleto' : 1, 
                                               'Fundamental Completo' : 2, 
                                               'Médio Completo' : 3, 
                                               'Superior Incompleto' : 4,
                                               'Superior Completo' : 5})
print(shapiro(X))

vif_data = pd.DataFrame()
vif_data['feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i)
                   for i in range(len(X.columns))]
print(vif_data)

#Complmento para VIF - Detecção de Multicolinearidade via HEATMAP
df = data
df.drop(['id_municipio', 'horas_cont', 'salario_mensal'], axis = 1, inplace = True)
srn.heatmap(data.corr(), annot = True, cmap='RdYlGn',square=True)

#----------------------- // ------------------------

#MODELO 2 CONSIDERANDO APENAS 44 HORAS

data2 = data.loc[data['horas_cont'] == 44]

horas_agrup = data2.groupby(['horas_cont']).size()
horas_agrup.plot.bar(color = 'gray')

cidade_agrup = data2.groupby(['centro_urb']).size()
cidade_agrup.plot.bar(color = 'gray')

sexo_agrup = data2.groupby(['sexo']).size()
sexo_agrup.plot.bar(color = 'gray')

raca_agrup = data2.groupby(['raca_cor']).size()
raca_agrup.plot.bar(color = 'gray')

esc_agrup = data2.groupby(['grau_instrucao']).size()
esc_agrup.plot.bar(color = 'gray') 

print(data2['salario_mensal'].describe())
srn.boxplot(data2['salario_mensal'])
srn.distplot(data2['salario_mensal'])

#regression model
model = sm.ols(formula = 'salario_mensal ~ grau_instrucao + idade + sexo + raca_cor + centro_urb', data = data2)
model_t = model.fit()
model_t.summary()

#------------------- // ----------------
#plots
srn.distplot(data['idade'], kde=True)
plt.hist(data['idade'], color = 'gray')
#normalidade de erros residuais
srn.distplot(model_t.resid, kde = True)
#multicolinearidade
data2 = data
data2.drop(['salario_mensal', 'id_municipio', 'horas_cont', 'salary_log', 'idade_2'], axis = 1, inplace = True)
data2.rename(columns = {'grau_instrucao' : 'grau_instr.'}, inplace = True)
srn.heatmap(data2.corr(), annot = True, cmap='RdYlGn',square=True)
#scatter
plt.scatter(data['idade'], data['salario_mensal'])
plt.scatter(data['centro_urb'], data['salario_mensal'])