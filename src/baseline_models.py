from collections import defaultdict
import numpy as np
from collections import Counter
from sklearn.linear_model import ElasticNet, ElasticNetCV, LassoCV, RidgeCV, LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV,StratifiedKFold
from sklearn.metrics import root_mean_squared_error, ndcg_score
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
from sklearn.kernel_ridge import KernelRidge
import pandas as pd 
import tqdm
from typing import Dict, List
import evaluation_metrics as evm
import utils
import argparse 
import yaml
import os
import sys
from lifelines.utils import concordance_index
from sklearn.feature_selection import SelectKBest, mutual_info_regression

def main(config:Dict):

	# Read in necessary file names and parameter settings from the config file
	training_screen, testing_screen, train_expr_file, train_resp_file, test_expr_file, test_resp_file, drug_file =\
		utils.unpack_parameters(config['DATA_PARAMS'])

	train_ccl_file, test_ccl_file, gene_sets =\
		utils.unpack_parameters(config['SELECTION_PARAMS'])

	model_names,alpha_min, alpha_max, num_alphas, auto_alphas, l1_ratios =\
		utils.unpack_parameters(config['MODEL_PARAMS'])

	num_bins, num_folds =\
		utils.unpack_parameters(config['TRAINING_PARAMS'])

	k_min, k_max, k_step, thresholds, testing_consistency =\
		utils.unpack_parameters(config["EVAL_PARAMS"])

	k_range = np.arange(k_min, k_max+1, k_step)
	rng = np.random.default_rng(seed=123450)
	
	
	# read in the list of drugs for testing
	drugs = utils.read_text_file(drug_file)
	
	

	
	if auto_alphas:
		alphas = None
	else:
		alphas = np.logspace(alpha_min,alpha_max,num_alphas)
	
	
	# load in the cell lines under consideration
	train_ccls = utils.read_text_file(train_ccl_file)
	test_ccls = utils.read_text_file(test_ccl_file)

	
	# Choose whether or not the testing should be done on new or overlapping ccls
	if testing_consistency:
		test_ccls = [x for x in test_ccls if x in train_ccls]
		# overlap = [x for x in train_ccls if x in test_ccls]
		
		# sys.exit()
		ev_type = "consistency"

	else:
		test_ccls = [x for x in test_ccls if x not in train_ccls]
		ev_type = "generalization"
	

	train_resp, train_expr = utils.fetch_data_set_pair(
		train_expr_file,train_resp_file)
	
	test_resp, test_expr = utils.fetch_data_set_pair(
		test_expr_file,test_resp_file)

	
	
	res_path = f"../results/{training_screen}_{testing_screen}/{ev_type}/"
	os.makedirs(res_path,exist_ok = True)


	# load in the genes that we want to keep
	keep_genes = utils.load_gene_sets(list(gene_sets.values()))
	

	# initialize dictionaries for storing results

	ranking_results = defaultdict(list)
	scalar_results = defaultdict(list)
	size_log = defaultdict(list)

	for drug in tqdm.tqdm(drugs):

		train_aacs = train_resp[train_resp['Drug']==drug]['AAC'].values
		thresh_to_cutoff = {}
		
		for thresh in thresholds:
			cutoff = np.round(np.percentile(train_aacs,100-thresh),4)
			thresh_to_cutoff[thresh] = float(cutoff)

	
		
		drug_list = [drug]

		response_subset = train_resp[train_resp['Drug']==drug]
		## ccls for which we have a training piint
		test_resp_ccls = pd.unique(test_resp[test_resp['Drug']==drug]['Cell_line'])
		ccl_subset = [ccl for ccl in train_ccls if ccl in pd.unique(response_subset['Cell_line'])]
		
		
		overlap = [ccl for ccl in test_ccls if ccl in ccl_subset]
		overlap = [ccl for ccl in overlap if ccl in test_resp_ccls]
		# keep_genes = gs_dict[gs]
	
		

		
		X_train, y_train = utils.build_data_set(
			train_expr, 
			response_subset,
			drug_list,
			ccl_subset,
			keep_genes)
		


		X_test, y_test = utils.build_data_set(
			test_expr,
			test_resp,
			drug_list,
			test_ccls,
			keep_genes)

		# log the size of the datasets
		size_log['Drug'].append(drug)
		size_log['N_train'].append(X_train.shape[0])
		size_log['N_test'].append(X_test.shape[0])

			
		for stratify in [True,False]:
			if stratify:
				y_cat = pd.qcut(y_train,q = num_bins,labels=range(num_bins),duplicates='drop')
				
				cv_info = [(i,j) for i,j in StratifiedKFold(num_folds).split(y_train,y_cat)]
				
			else:
				cv_info = num_folds
			
			
			for model_name in (pbar := tqdm.tqdm(model_names,leave=False)):
				
				pbar.set_description(f"Fitting {model_name}")
				
				if model_name=="ElasticNet":
					model = ElasticNetCV(
						l1_ratio = l1_ratios,
						alphas=alphas,
						cv = cv_info,
						precompute=True,
						max_iter = 10**4,
						selection = 'random',
						n_jobs = -1)
					model.fit(X_train,y_train)
					preds = model.predict(X_test)
				
				elif model_name == "Lasso":
					model = LassoCV(
						alphas = alphas, 
						cv = cv_info,
						max_iter = 10**6,
						n_jobs = -1)
					model.fit(X_train,y_train)
					preds = model.predict(X_test)
				
				elif model_name == "Ridge":
					model = RidgeCV(
						alphas = alphas, 
						cv = cv_info)
					model.fit(X_train,y_train)
					preds = model.predict(X_test)
				
				elif model_name == "RandomForest":
					regr = RandomForestRegressor(n_jobs = -1)
					param_grid = {
						'max_features':["sqrt","log2",1.0],
						'n_estimators':[1,5,10,25,50,75,100]
						}
					model = GridSearchCV(regr,param_grid,n_jobs=-1,cv=cv_info)
					model.fit(X_train,y_train)
					preds = model.predict(X_test)
				
				elif model_name == "UniLasso":
					# print("in unilasso")
					np.savetxt("./train_data.csv",X_train,delimiter=",",fmt="%.18f",newline="\n")
					np.savetxt("./train_resp.csv",y_train,delimiter=",",fmt="%.18f",newline="\n")
					np.savetxt("./test_data.csv",X_test,delimiter=",",fmt="%.18f",newline="\n")
					os.system("Rscript run.uniLasso.R train_data.csv train_resp.csv test_data.csv preds.csv gaussian")
					df = pd.read_csv("./preds.csv",index_col=0)
					preds = df["preds"].values
				elif model_name == "KRR":
					regr = KernelRidge()
					param_grid = {'alpha':np.linspace(0.01,5,100)}
					model =GridSearchCV(regr,param_grid,n_jobs=-1, cv=cv_info)
					model.fit(X_train,y_train)
					preds = model.predict(X_test)
				elif model_name == "Univariable":
					model = Pipeline([
						('select',SelectKBest(mutual_info_regression,k=1)),
						('predict',LinearRegression())
						])
					model.fit(X_train,y_train)
					preds = model.predict(X_test)
				
				elif model_name == "Random":
					preds = rng.uniform(size=y_test.shape[0])
					# print(preds)
		
				elif model_name == "GroundTruth":
					
					_, y_test = utils.build_data_set(
						test_expr,
						test_resp,
						drug_list,
						overlap,
						keep_genes)

					_, preds = utils.build_data_set(
						train_expr, 
						response_subset,
						drug_list,
						overlap,
						keep_genes)
					



					
				# print(f"\nModel {model_name} with {len(y_test)} test points and {len(preds)} pred points")
				rmse = root_mean_squared_error(y_test,preds)
				
				ci = float(concordance_index(y_test,preds))
				
				corr,_ = pearsonr(y_test,preds)
				
				_k_range = [x for x in k_range if x<X_test.shape[0]]
				
				for k in _k_range:
					# scalar_results['Gene Set'].append(gs)		
					perm_prec = evm.perm_precision(y_test,preds,k=k)
					ndcg = evm.ndcg_at_k(y_test, preds, k, method = 0)
					
					scalar_results['Drug'].append(drug)
					scalar_results['Stratified'].append(stratify)
					scalar_results['Model'].append(model_name)
					scalar_results['RMSE'].append(rmse)
					scalar_results['Concordance'].append(ci)
					scalar_results['PermPrec'].append(perm_prec)
					# scalar_results["NDCG0"].append(ndcg0)
					# scalar_results["NDCG1"].append(ndcg1)
					scalar_results["NDCG"].append(ndcg)
					# scalar_results["NDCG_shifted"].append(ndcg_shift)
					scalar_results["Rsquared"].append(corr)
					# scalar_results["TP"].append(trunc_perm)
					scalar_results['K'].append(k)

				
				
				for thresh in thresholds:
					y_test_ordinal = utils.make_ordinal(y_test, cutoff = thresh_to_cutoff[thresh])

					# print(np.sum(y_test_ordinal))
					
					for k in _k_range:
						# ranking_results["Gene Set"].append(gs)
						
						prec = evm.precision_at_k(y_test_ordinal,preds, k)
						avg_prec = evm.average_precision_at_k(y_test_ordinal,preds,k)
						rr = evm.reciprocal_rank(y_test_ordinal,preds)

						ranking_results['Drug'].append(drug)
						ranking_results['Ordinal Type'].append("binary")
						ranking_results['Stratify'].append(stratify)
						ranking_results['Model'].append(model_name)
						ranking_results['Threshold'].append(thresh)
						ranking_results['Precision'].append(prec)
						ranking_results['Average Precision'].append(avg_prec)
						ranking_results['RR'].append(rr)
						ranking_results['K'].append(k)

	scalar_results = pd.DataFrame(scalar_results)
	ranking_results = pd.DataFrame(ranking_results)
	size_log = pd.DataFrame(size_log)

	scalar_results.to_csv(f"{res_path}scalar_metrics.csv",index=False)
	ranking_results.to_csv(f"{res_path}ranking_metrics.csv",index=False)
	size_log.to_csv(f"{res_path}sizes.csv",index=False)
	






	



if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(
		prog = "Baseline models for DeepCINet.",
		description = "")

	parser.add_argument("-config", help = "The configuration file for the experiment.")

	args = parser.parse_args()
	with open(args.config) as config_file:
		config = yaml.safe_load(config_file)

	main(config)