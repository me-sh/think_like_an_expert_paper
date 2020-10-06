#! /bin/env bash

#SBATCH -J 'run_feat'
#SBATCH -o /mnt/bucket/people/meshulam/slurm_logs/%j.out
#SBATCH -p all
#SBATCH -t 150



module load fsl
echo "inputs:"
echo "epi original directory: $1"
echo "epi original file name only: $2"
#echo "epi file in: ${1}/${2}.nii.gz"
echo "epi new directory (=feat directory): $3"
echo "path for fsf file: $4"
echo "subject anatomy file full path: $5"
echo "fsf template file full path: $6"
echo "TR length in seconds: $7"
echo "Smoothing FWHM: $8"
echo "Highpass threshold: $9"
echo "MNI brain: ${10}"


#prep inputdir ###INPUTDIR
epi_file_in=${1}/${2}.nii.gz
#prep outputdir ###OUTPUTDIR###
out_dir=${3}
# fsf output dir
fsf_dir=${4}
#prep anatdir ###ANATDIR###
anat_file_in=${5}
#prep template file
template_file=${6}
# prep nvols ###NVOLS### 
nvols=`fslnvols ${epi_file_in}`
#prep TR length ###TR_IN_SECS###
tr_length=${7}
#prep smoothing ###SMOOTH_MM###
smoothing_mm=${8}
#prep highpass ###HIGHPASS###
highpass_threshold=${9}
#prep standard brain full file
mni_brain=${10}

# replace strings in fsf file and write to feat dir
sed -e "s@###INPUTDIR###@${epi_file_in}@g" \
	-e "s@###ANATDIR###@$anat_file_in@g" \
	-e "s@###NVOLS###@$nvols@g" \
	-e "s@###TR_IN_SECS###@$tr_length@g" \
	-e "s@###SMOOTH_MM###@$smoothing_mm@g" \
	-e "s@###HIGHPASS###@$highpass_threshold@g" \
	-e "s@###OUTPUTDIR###@$out_dir@g" \
	-e "s@###MNIBRAIN###@$mni_brain@g" \
	${template_file} > ${fsf_dir}/${2}.fsf
echo done

echo "run feat"
feat ${fsf_dir}/${2}.fsf
echo "done"

echo "done and done"
