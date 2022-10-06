![NLP_-_Classificador_de_Texto (1)](https://user-images.githubusercontent.com/105673165/194183096-41a7c8ec-5f14-4298-a276-f7ebeb601c9f.png)

## Desafio 1 Hand Talk - classificação de frases por setor
Dada uma base de dados com frases classificadas em contexto/setores (atividades econômicas) específicos, seu objetivo, caso deseje aceitar, é criar um modelo de classificação para classificar frases nos setores anotados.

Logo abaixo temos a base de dados que você deverá usar para treinar seu modelo. Ela contém 521 frases classificadas entre os setores: finanças, educação, indústrias, varejo, orgão público.

Importante avisar que na base de dados existem frases classificadas com um único ou múltiplos setores. O caso de múltiplos setores ocorre devido a frase poder possuir um vocabulário cujo os termos pertencem a setores diferentes. Exemplo:

"Curso de Técnico em Segurança do Trabalho por 32x R$ 161,03".

É uma frase que poderia ser classificada nos setores educação e finanças.

Logo, seu modelo deve tratar a possibilidade da frase possuir multiplas classificações!

*---------------------------------------------------------------------------------------------------------------------------------*

Este projeto foi o primeiro que realizei dos desafios. Como sempre, trabalhar com strings pode realmente ser desafiador.

Comecei buscando algumas referências de projetos desenvolvidos na mesma linha, e encontrei algumas coisas interessantes.

O primeiro passo era analisar / explorar os dados e entender o que eu tinha em mãos.

![image](https://user-images.githubusercontent.com/105673165/194183684-e7ea0f5c-a82b-4481-b589-cab5f5736151.png)

Como pode-se observar, as frases não possuem nenhum padrão (ou seja, vão precisar de tratamento), e as tags são strings, então logo já penso em fazer um HotEncoding dessas tags. E é o que eu fiz.
Também removi acentos, "letras sozinhas" e espaços múltiplos das frases.

![image](https://user-images.githubusercontent.com/105673165/194185116-629c7b51-f08e-4d09-ac6b-d7792a304ac6.png)

Aqui considerei que o tratamento dos dados estava finalizado.

Outro passo necessário era vetorizar as frases para que pudéssemos utilizar no treino do modelo ("feature engineering"). Para isso, utilizei o TFIDF Vectorizer do sk-learn, que transformou as frases:

![image](https://user-images.githubusercontent.com/105673165/194186694-e9029d2a-2b4c-4c8f-a7f5-fa317e5f493e.png)

Então parti para a construção dos modelos.

Em minha primeira tentativa, implementei uma rede neural mais "chique" com o TensorFlow, porém não consegui classificar efetivamente as frases. Então retornei para alguns modelos "mais simples", também por conta da baixa quantidade de dados que tinha.

Utilizei alguns métodos (do **scikit-multilearn**) de transformação do problema de multi-label das frases:
- **BinaryRelevance** - Transforma o problema de classificação multi-label (com N labels) em (N) problemas separados de classificação binária;
- **Classifier Chains** - Para N labels ele treina N classificadores ordenados de acordo com a Bayesian Chain Rule;
- **LabelPowerset** - Transforma um problema multi-label em um problema multi-classe com 1 classificador multi-classe treinado em todas as combinações de labels diferentes (consideradas as classes) encontradas nos dados.

Para cada transformação, construí um modelo baseado em **Multinomial Naive Bayes**.

BinaryRelevance e Classifier Chains não performaram nada bem.
#### BR:

![image](https://user-images.githubusercontent.com/105673165/194188921-cc2cc52a-1a67-4ae1-a449-1dd61a80b621.png)

#### CC:

![image](https://user-images.githubusercontent.com/105673165/194188978-0804bf19-bcad-4f5d-9094-166aaa8377a7.png)

Enquanto isso, o modelo utilizando **Label Powerset** trouxe resultados razoáveis, porém não classificava as frases em mais de uma label, NUNCA!

![image](https://user-images.githubusercontent.com/105673165/194189086-1d5d95a4-9296-4644-a26a-2fa99818fac8.png)

Esse modelo (MultinomialNB + Label Powerset) foi o primeiro que considerei "semi-funcional". E por conta do tempo parti para a resolução dos outros desafios.

Posteriormente, retornei à este desafio com a ideia de mudar o algoritmo utilizado no modelo.

Então desenvolvi novamente, 3 modelos (com as três mesmas transformações) porém utilizando o **Gaussian Naive Bayes**, e os resultados foram consideravelmente melhores para todos os modelos.

Os modelos com BR e CC melhoraram consideravelmente suas métricas, e o modelo com Label Powerset melhorou bem pouco sua acurácia, mas melhorou.
Porém, agora todos os modelos estavam classificando algumas frases com mais de um label! Então agora o modelo com Label Powerset + Gaussian NB ficou um pouco melhor, mas 100% funcional.

![image](https://user-images.githubusercontent.com/105673165/194189487-0e4ff7ef-a823-4a35-b3d4-ce1b662531cb.png)

Salvando o modelo, podemos carregá-lo para fazer previsões sem treiná-lo novamente.

![image](https://user-images.githubusercontent.com/105673165/194190118-7bc6c31e-10ac-451c-b70b-abcc64a0db27.png)

Também, podemos construir uma pipeline de dados que inclui o vetorizador e o modelo, para uma utilização mais prática!



## Link References:

#### Multi Label Text Classification with Scikit-Learn - Susan Li:

https://towardsdatascience.com/multi-label-text-classification-with-scikit-learn-30714b7819c5

#### Multi-Label Classification with Python and Scikit-Multilearn - JCharis Jesse:

https://github.com/Jcharis/Python-Machine-Learning/blob/master/Multi_Label_Text_Classification_with_Skmultilearn/Multi-Label%20Classification%20with%20Python%20and%20Scikit-Multilearn-.ipynb
