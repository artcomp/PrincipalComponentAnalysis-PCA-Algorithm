from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from numpy import linalg as LA
from sklearn.decomposition import PCA
from kmeans import *

file = r'DBMS-Performance-Monitor-Log.xls'
df = pd.read_excel(file)

# remove dados não necessarios
df = df.drop('Index', axis=1)
df = df.drop('TR ID ', axis=1)

def letraA():
	descriptive_stats_df = df.describe(include='all')
	print(descriptive_stats_df)


def letraB():
	################### HISTOGRAMAS

	plt.hist(df['CPU'], bins=10)
	plt.title('Histograma da CPU')
	plt.savefig('histogram-CPU.png', format='png')
	plt.clf()

	plt.hist(df['Disk 1'], bins=10)
	plt.title('Histograma do Disk 1')
	plt.savefig('histogram-disk1.png', format='png')
	plt.clf()

	plt.hist(df['Disk 2'], bins=10)
	plt.title('Histograma do Disk 2')
	plt.savefig('histogram-disk2.png', format='png')
	plt.clf()

	################### CDF

	plt.hist(df['CPU'], cumulative=True)
	plt.title('CDF da CPU')
	plt.savefig('cfd-CPU.png', format='png')
	plt.clf()

	plt.hist(df['Disk 1'], cumulative=True)
	plt.title('CDF do Disk 1')
	plt.savefig('cfd-disk1.png', format='png')
	plt.clf()

	plt.hist(df['Disk 2'], cumulative=True)
	plt.title('CDF do Disk 2')
	plt.savefig('cfd-disk2.png', format='png')
	plt.clf()



	################### BOX PLOT
	plt.figure()
	df.boxplot()
	plt.savefig('boxplot.png', format='png')


def applyPercentage(l):
	return list( map(lambda x: x * 100, l) )

def printDataFrame(df):
	with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
		print(df)

def letraC():
	# PCA
	normalized_data_frame = (df-df.mean())/df.std()
	descriptive_stats_normalized_df = normalized_data_frame.describe(include='all')

	matrix_de_correlacao = normalized_data_frame.corr()

	auto_valores, auto_vetores = LA.eig(matrix_de_correlacao)
	
	multiplicacao_auto_vetor_dados_normalizados = np.matmul(normalized_data_frame, auto_vetores)

	multiplicacao_auto_vetor_dados_normalizados.columns = ['Y CPU', 'Y Disk 1', 'Y Disk 2']
	
	# comentado apenas por causa da largura de exibicao do terminal
	# normalized_data_frame['Y CPU'] = multiplicacao_auto_vetor_dados_normalizados['Y CPU']
	# normalized_data_frame['Y Disk 1'] = multiplicacao_auto_vetor_dados_normalizados['Y Disk 1']
	# normalized_data_frame['Y Disk 2'] = multiplicacao_auto_vetor_dados_normalizados['Y Disk 2']

	normalized_data_frame['CPU ^ 2'] = multiplicacao_auto_vetor_dados_normalizados['Y CPU'] ** 2
	normalized_data_frame['Disk 1 ^ 2'] = multiplicacao_auto_vetor_dados_normalizados['Y Disk 1'] ** 2
	normalized_data_frame['Disk 2 ^ 2'] = multiplicacao_auto_vetor_dados_normalizados['Y Disk 2'] ** 2

	cpu_square_sum = normalized_data_frame['CPU ^ 2'].sum()
	disk_1_square_sum = normalized_data_frame['Disk 1 ^ 2'].sum()
	disk_2_square_sum = normalized_data_frame['Disk 2 ^ 2'].sum()

	total_sum = cpu_square_sum + disk_1_square_sum + disk_2_square_sum
	cpu_impact = cpu_square_sum / total_sum
	disk_1_impact = disk_1_square_sum / total_sum
	disk_2_impact = disk_2_square_sum / total_sum



	components_stats = [cpu_impact, disk_1_impact, disk_2_impact]
	percentages = applyPercentage(components_stats)	

	# plot impacto de cada recurso no sistema
	plt.clf()
	plt.bar(['CPU', 'Disk 1', 'Disk 2'], percentages )
	plt.show()
	plt.xlabel('Recurso')
	plt.ylabel('Porcentagem de impacto no sistema')
	plt.title('Impacto de cada recurso no sistema')
	plt.savefig('resources-impact.png', format='png')
	plt.clf()
	

	# scatterplot CPU x disk 2 (fatores de maior impacto)
	# plt.clf()
	# plt.scatter(normalized_data_frame['CPU'], normalized_data_frame['Disk 2'] )
	# plt.xlabel('CPU')
	# plt.ylabel('Disk 2')
	# plt.title('Scatter Plot')
	# plt.savefig('scatter-plot.png', format='png')
	# plt.clf()

def letraD():
	print('''\n\nLetra (d)\n\tO fator de maior impacto é a CPU, representado 80%.
		 	\n\tAssim, pode ser unicamente utilizado para a clusterização dos dados.\n\n\n''')


def scyPCA(plot):

	pca = PCA(n_components=3)
	pca.fit(df)
	percentages = pca.explained_variance_ratio_
	print('Resultado do PCA : \n\n\tCPU : %f\n\tDisk 1 : %f\n\tDisk 2 : %f' % (percentages[0],percentages[1],percentages[2]))

	if plot != 0:
		plt.clf()
		plt.bar(['CPU', 'Disk 1', 'Disk 2'], applyPercentage(percentages) )
		plt.show()
		plt.xlabel('Recurso')
		plt.ylabel('Porcentagem de impacto no sistema')
		plt.title('Impacto de cada recurso no sistema')
		plt.savefig('resources-impact.png', format='png')
		plt.clf()


def main():


	# pca scykitlearn
	scyPCA(plot=0)

	num_clusters = [x for x in range(3,190)]

	data = [ [a,b] for a,b in zip(df['CPU'],(df['Disk 1'] + df['Disk 2']) ) ]

	cv_s = list( map( lambda x: clusterization(data , x, 0), num_clusters) ) 
	
	ctr = 1
	if ctr:
		plt.plot(num_clusters,cv_s)
		plt.xlabel('Número de clusters (K)')
		plt.ylabel('Coeficiente de Variação (CV)')
		plt.title('Número ideal de clusters ')
		plt.show()
		

	# ideal_k = 3
	# cv_ideal = clusterization(data, ideal_k, 1)

	
if __name__ == '__main__':
	main()