prediction = classify(net,featuresTrain);
figure
cm = confusionchart(categorical(labelsTrain),prediction,'title','Training Accuracy');
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';