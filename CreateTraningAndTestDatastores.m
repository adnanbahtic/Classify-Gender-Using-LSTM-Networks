url = 'http://ssd.mathworks.com/supportfiles/audio/commonvoice.zip';
downloadFolder = tempdir;
dataFolder = fullfile(downloadFolder,'commonvoice');
unzip(url,downloadFolder)

loc = fullfile(dataFolder);
adsTrain = audioDatastore(fullfile(loc,'train'),'IncludeSubfolders',true);
metadataTrain = readtable(fullfile(fullfile(loc,'train'),"train.tsv"),"FileType","text");
adsTrain.Labels = metadataTrain.gender;

adsValidation = audioDatastore(fullfile(loc,'validation'),'IncludeSubfolders',true);
metadataValidation = readtable(fullfile(fullfile(loc,'validation'),"validation.tsv"),"FileType","text");
adsValidation.Labels = metadataValidation.gender;

countEachLabel(adsTrain)
countEachLabel(adsValidation)


reduceDataset = false;
if reduceDataset
    adsTrain = splitEachLabel(adsTrain,round(numel(adsTrain.Files) / 2 / 20));
    adsValidation = splitEachLabel(adsValidation,20);
end