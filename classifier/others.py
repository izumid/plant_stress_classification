
def logTime(start,model,save_path):
	end = datetime.now()
	message = (f"{model}',duration (hhmmssms), {str((end-start))}\n")
	with open(os.path.join(save_path,'log_times.txt'), 'a') as log_file: log_file.write(message)


def save_csv(result,model_name,path_destination,file_name,start,backup,log):
	start_string = start.strftime("%Y%m%d_%Hh%Mm%Ss")
	r_root = Path(path_destination).parents[0]
	path_backup = os.path.join(r_root,"backup")
	file_name_zip = start_string+'_'+file_name+"_"+model_name+".zip"
	file_name_csv = file_name_zip.replace("zip","csv")
	path_csv_file = os.path.join(path_destination,file_name_csv)
	path_zip = os.path.join(path_backup,file_name_zip)
	file_path = os.path.join(path_destination, file_name_csv)
	path_log = os.path.join(path_backup,"log_times.zip")
	file_log_name = "log_times.txt"
	title_log = file_name.replace("_",',').title()+","

	result_dataframe = pd.DataFrame(result, columns=["Stimulus","Flow","Window","TrainAccuracy","Accuracy","Precision","Recall","F1Score"])
	result_dataframe["Model"] = model_name 

	if model_name == "DUM": result_dataframe = result_dataframe.drop_duplicates(keep="first")

	if not os.path.exists(path_destination): os.makedirs(path_destination)

	if os.path.isfile(path_csv_file): result_dataframe.to_csv(path_csv_file,float_format='%.4f',index=False,mode='a',header=False)
	else: result_dataframe.to_csv(path_csv_file,float_format='%.4f',index=False)

	if not os.path.exists(path_backup): os.makedirs(path_backup)

	#zip files
	if backup == True: 
		with zipfile.ZipFile(path_zip, 'w',compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file: zip_file.write(file_path, arcname=file_name_csv)
			
	#zip log
	if log == True: 
		logTime(start,title_log+model_name,r_root)
		with zipfile.ZipFile(path_log, 'w',compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file: zip_file.write(os.path.join(r_root,file_log_name), arcname=file_log_name)

