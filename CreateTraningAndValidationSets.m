[~,adsInfo] = read(adsTrain);
Fs = adsInfo.SampleRate;
extractor.SampleRate = Fs;
extractor.Window = hamming(round(0.03*Fs),"periodic");
extractor.OverlapLength = round(0.02*Fs);

if ~isempty(ver('parallel')) && ~reduceDataset
    pool = gcp;
    numPar = numpartitions(adsTrain,pool);
else
    numPar = 1;
end

labelsTrain = [];
featureVectors = {};


parfor ii = 1:numPar
    

    subds = partition(adsTrain,numPar,ii);
    
    featureVectorsInSubDS = {};
    segmentsPerFile = zeros(numel(subds.Files),1);
    
    for jj = 1:numel(subds.Files)
        
        % 1. Read in a single audio file
        audioIn = read(subds);
        
        % 2. Determine the regions of the audio that correspond to speech
        speechIndices = detectSpeech(audioIn,Fs);
        
        % 3. Extract features from each speech segment
        segmentsPerFile(jj) = size(speechIndices,1);
        features = cell(segmentsPerFile(jj),1);
        for kk = 1:size(speechIndices,1)
            features{kk} = ( extract(extractor,audioIn(speechIndices(kk,1):speechIndices(kk,2))) )';
        end
        featureVectorsInSubDS = [featureVectorsInSubDS;features(:)];
        
    end
    featureVectors = [featureVectors;featureVectorsInSubDS];
    
   
    repedLabels = repelem(subds.Labels,segmentsPerFile);
    labelsTrain = [labelsTrain;repedLabels(:)];
end

allFeatures = cat(2,featureVectors{:});
allFeatures(isinf(allFeatures)) = nan;
M = mean(allFeatures,2,'omitnan');
S = std(allFeatures,0,2,'omitnan');
featureVectors = cellfun(@(x)(x-M)./S,featureVectors,'UniformOutput',false);
for ii = 1:numel(featureVectors)
    idx = find(isnan(featureVectors{ii}));
    if ~isempty(idx)
        featureVectors{ii}(idx) = 0;
    end
end

[featuresTrain,trainSequencePerSegment] = HelperFeatureVector2Sequence(featureVectors,featureVectorsPerSequence,featureVectorOverlap);
labelsTrain = repelem(labelsTrain,[trainSequencePerSegment{:}]);
labelsTrain = categorical(labelsTrain);

labelsValidation = [];
featureVectors = {};
valSegmentsPerFile = [];
parfor ii = 1:numPar
    subds = partition(adsValidation,numPar,ii);
    featureVectorsInSubDS = {};
    valSegmentsPerFileInSubDS = zeros(numel(subds.Files),1);
    for jj = 1:numel(subds.Files)
        audioIn = read(subds);
        speechIndices = detectSpeech(audioIn,Fs);
        numSegments = size(speechIndices,1);
        features = cell(valSegmentsPerFileInSubDS(jj),1);
        for kk = 1:numSegments
            features{kk} = ( extract(extractor,audioIn(speechIndices(kk,1):speechIndices(kk,2))) )';
        end
        featureVectorsInSubDS = [featureVectorsInSubDS;features(:)];
        valSegmentsPerFileInSubDS(jj) = numSegments;
    end
    repedLabels = repelem(subds.Labels,valSegmentsPerFileInSubDS);
    labelsValidation = [labelsValidation;repedLabels(:)];
    featureVectors = [featureVectors;featureVectorsInSubDS];
    valSegmentsPerFile = [valSegmentsPerFile;valSegmentsPerFileInSubDS];
end

featureVectors = cellfun(@(x)(x-M)./S,featureVectors,'UniformOutput',false);
for ii = 1:numel(featureVectors)
    idx = find(isnan(featureVectors{ii}));
    if ~isempty(idx)
        featureVectors{ii}(idx) = 0;
    end
end

[featuresValidation,valSequencePerSegment] = HelperFeatureVector2Sequence(featureVectors,featureVectorsPerSequence,featureVectorOverlap);
labelsValidation = repelem(labelsValidation,[valSequencePerSegment{:}]);
labelsValidation = categorical(labelsValidation);