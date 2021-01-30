[audioIn,Fs] = audioread('maleSpeech.flac');
labels = {'male'};

timeVector = (1/Fs) * (0:size(audioIn,1)-1);
figure
plot(timeVector,audioIn)
ylabel("Amplitude")
xlabel("Time (s)")
title("Sample Audio")
grid on

sound(audioIn,Fs)
speechIndices = detectSpeech(audioIn,Fs);

extractor = audioFeatureExtractor( ...
    "SampleRate",Fs, ...
    "Window",hamming(round(0.03*Fs),"periodic"), ...
    "OverlapLength",round(0.02*Fs), ...
    ...
    "gtcc",true, ...
    "gtccDelta",true, ...
    "gtccDeltaDelta",true, ...
    ...
    "SpectralDescriptorInput","melSpectrum", ...
    "spectralCentroid",true, ...
    "spectralEntropy",true, ...
    "spectralFlux",true, ...
    "spectralSlope",true, ...
    ...
    "pitch",true, ...
    "harmonicRatio",true);

featureVectorsSegment = {};
for ii = 1:size(speechIndices,1)
    featureVectorsSegment{end+1} = ( extract(extractor,audioIn(speechIndices(ii,1):speechIndices(ii,2))) )';
end
numSegments = size(featureVectorsSegment)

[numFeatures,numFeatureVectorsSegment1] = size(featureVectorsSegment{1})

labels = repelem(labels,size(speechIndices,1))

featureVectorsPerSequence = 20;
featureVectorOverlap = 5;
hopLength = featureVectorsPerSequence - featureVectorOverlap;

idx1 = 1;
featuresTrain = {};
sequencePerSegment = zeros(numel(featureVectorsSegment),1);
for ii = 1:numel(featureVectorsSegment)
    sequencePerSegment(ii) = max(floor((size(featureVectorsSegment{ii},2) - featureVectorsPerSequence)/hopLength) + 1,0);
    idx2 = 1;
    for j = 1:sequencePerSegment(ii)
        featuresTrain{idx1,1} = featureVectorsSegment{ii}(:,idx2:idx2 + featureVectorsPerSequence - 1);
        idx1 = idx1 + 1;
        idx2 = idx2 + hopLength;
    end
end

