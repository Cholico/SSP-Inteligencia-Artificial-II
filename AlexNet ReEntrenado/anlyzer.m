%% Paso 1: Reentrenar AlexNet con un Conjunto de Datos Personalizado

% Cargar la red preentrenada AlexNet
net = alexnet;

% Ver la arquitectura de la red
analyzeNetwork(net);

% Cargar un conjunto de datos de imagen
imageFolder = fullfile('images/');
imds = imageDatastore(imageFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Dividir los datos en conjuntos de entrenamiento y prueba
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');

% Redimensionar las imágenes a 227x227 (entrada esperada por AlexNet)
inputSize = net.Layers(1).InputSize;
augmentedTrainSet = augmentedImageDatastore(inputSize(1:2), imdsTrain);
augmentedTestSet = augmentedImageDatastore(inputSize(1:2), imdsTest);

% Extraer las capas de la red preentrenada, excluyendo las últimas tres capas
layersTransfer = net.Layers(1:end-3);

% Número de clases (ajustar según tu dataset)
numClasses = numel(categories(imdsTrain.Labels));

% Reemplazar la última capa totalmente conectada y la capa de salida
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
    softmaxLayer
    classificationLayer];

% Configurar las opciones de entrenamiento
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 64, ...
    'MaxEpochs', 6, ...
    'InitialLearnRate', 1e-4, ...
    'ValidationData', augmentedTestSet, ...
    'ValidationFrequency', 10, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Entrenar la red
netTransfer = trainNetwork(augmentedTrainSet, layers, options);

% Evaluar la red en el conjunto de prueba
YPred = classify(netTransfer, augmentedTestSet);
YTest = imdsTest.Labels;

accuracy = mean(YPred == YTest);
disp(['Accuracy: ' num2str(accuracy)]);

% Guardar el modelo entrenado
save('modelo.mat', 'netTransfer');

%% Paso 2: Usar la Red Reentrenada para Clasificación en Tiempo Real con Webcam

% Cargar el modelo entrenado
loadedModel = load('modelo.mat');
netTransfer = loadedModel.netTransfer;

% Configurar la webcam
cam = webcam; % Conectar a la primera cámara disponible

% Redimensionar las imágenes a 227x227 (entrada esperada por AlexNet)
inputSize = netTransfer.Layers(1).InputSize;

% Bucle para capturar y clasificar imágenes en tiempo real
while true
    % Capturar una imagen de la webcam
    img = snapshot(cam);
    
    % Redimensionar la imagen a 227x227
    img = imresize(img, [227 227]);
    
    % Hacer la predicción
    label = classify(netTransfer, img);
    
    % Mostrar la imagen y la etiqueta
    imshow(img);
    title(char(label));
    
    % Pausa para que la imagen sea visible y evitar un bucle demasiado rápido
    pause(0.1);
end
