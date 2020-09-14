# How to run: execute `bash run.sh <team_name>`

# Use team name from first argument ($1 = teamName)
teamName=$1
link=$2

# Download file from MEGA or from HTTPS link with wget
if [ ! -z $link ]; then
    echo "Custom link"
    if [[ $link == *"mega.nz"* ]]; then
        echo "Mega file found"
        singularity exec mega.sif mega-get -m $link ./images/$teamName.sif
    elif [[ $link == *"drive.google"* ]]; then
        echo "Google drive file found"
        singularity exec gdown.sif $link /home/victor_campello/mnms/images/$teamName.sif
    else
        echo "Other type of file found"
        wget -O /home/victor_campello/mnms/images/$teamName.sif $link
    fi
else
    echo "File in MEGA"
    # singularity exec mega.sif mega-get -m Singularity/$teamName/$teamName.sif /home/victor_campello/mnms/images/$teamName.sif
    singularity exec mega.sif mega-get -m Singularity/$teamName/${teamName}.sif ./images/$teamName.sif
fi


# Substitute it in the template job
cat example_job.template.sh | sed -e 's/\$1/'$teamName'/g' > jobs/$teamName.sh
exit 1
# Run the resulting job
sbatch jobs/$teamName.sh
