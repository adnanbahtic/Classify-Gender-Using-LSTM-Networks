load('genderIDNet.mat', 'genderIDNet', 'M', 'S');

[audioIn, Fs] = audioread('maleSpeech.flac');
sound(audioIn, Fs)

boundaries = detectSpeech(audioIn, Fs);
audioIn = audioIn(boundaries(1):boundaries(2));

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

features = extract(extractor, audioIn);
features = (features.' - M)./S;

gender = classify(genderIDNet, features)

[audioIn, Fs] = audioread('femaleSpeech.flac');
sound(audioIn, Fs)

boundaries = detectSpeech(audioIn, Fs);
audioIn = audioIn(boundaries(1):boundaries(2));

features = extract(extractor, audioIn);
features = (features.' - M)./S;

classify(genderIDNet, features)