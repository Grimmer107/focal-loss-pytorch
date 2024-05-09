## Focal Loss

Focal loss was introduced in 2017. It is particularly helpful when dealing 
with data with class imbalances. Another example, is in the case of Object 
Detection when most pixels are usually background and only very few pixels
inside an image sometimes have the object of interest.

What Focal Loss does is that it makes it easier for the model to predict things 
without being 80-100% sure that this object is "something". In simple words, 
giving the model a bit more freedom to take some risk when making predictions.
This is particularly important when dealing with highly imbalanced datasets 
because in some cases (such as cancer detection), we really need to model 
to take a risk and predict something even if the prediction turns out to be
a False Positive.

![image](https://github.com/Grimmer107/focal-loss-pytorch/assets/111426368/e4060f0b-e899-43a6-9891-42a8e61b79ca)

In the graph above, the "blue" line represents the Cross Entropy Loss. The X-axis
or 'probability of ground truth class' (let's call it pt for simplicity) is the 
probability that the model predicts for the ground truth object. As an example, 
let’s say the model predicts that something is a bike with probability 0.6 and it
actually is a bike. The pt in this case pt is 0.6. Also, consider the same example but
this time the object is not a bike. Then pt is 0.4 because ground truth here is 0 and
probability that the object is not a bike is 0.4 (1-0.6).
The Y-axis is simply the loss value given pt.

As can be seen from the image, when the model predicts the ground truth with a probability
of 0.6, the Cross Entropy Loss is still somewhere around 0.5. Therefore, to reduce the loss,
our model would have to predict the ground truth label with a much higher probability. In 
other words, Cross Entropy Loss asks the model to be very confident about the ground truth 
prediction. This in turn can actually impact the performance negatively. 
The Deep Learning model can actually become overconfident and therefore, the model wouldn’t
generalize well.

As can be seen from the graph, you can compare Focal Loss with Cross Entropy, using Focal Loss 
with gemma > 1 reduces the loss for "well-classified examples" or examples when the model predicts
the right thing with probability > 0.5 whereas, it increases loss for "hard-to-classify examples" when
the model predicts with probability < 0.5. Therefore, it turns the models attention towards the rare class
in case of class imbalance.

The Focal Loss is mathematically defined as:

![image](https://github.com/Grimmer107/focal-loss-pytorch/assets/111426368/b9868904-a99a-4c28-b712-5c02fa29709e)


The gemma controls the shape of the curve. The higher the value of γ gemma, the lower the loss for well-classified
examples, so we could turn the attention of the model more towards hard-to-classify examples. Having
higher γ extends the range in which an example receives low loss. Also, when γ = 0, this equation is 
equivalent to Cross Entropy Loss.

Another way, apart from Focal Loss, to deal with class imbalance is to introduce weights. Give high weights
to the rare class and small weights to the dominating or common class. These weights are referred to as alpha α.

Adding these weights does help with class imbalance however, the focal loss paper reports: The large class
imbalance encountered during training of dense detectors overwhelms the cross entropy loss. Easily classified
negatives comprise the majority of the loss and dominate the gradient. While α balances the importance of 
positive/negative examples, it does not differentiate between easy/hard examples.

What the authors are trying to explain is this: Even when we add α, while it does add different weights to 
different classes, thereby balancing the importance of positive/negative examples - just doing this in most 
cases is not enough. What we also want to do is to reduce the loss of easily-classified examples because otherwise
these easily-classified examples would dominate our training.





