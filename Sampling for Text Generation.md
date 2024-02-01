# Sampling for Text Generation

## 1. Sampling
 Với một đầu vào, mạng nơ-ron tạo ra đầu ra bằng cách tính toán xác suất của tất cả các giá trị có thể có. Đối với một trình phân loại, các giá trị có thể có là các lớp có sẵn. Ví dụ: nếu một mô hình được đào tạo để phân loại xem một email có phải là thư rác hay không thì chỉ có hai giá trị có thể có: thư rác và không phải thư rác. Mô hình tính toán xác suất của từng giá trị trong số hai giá trị này, giả sử là thư rác là 90% và không phải thư rác là 10%.
Để tạo mã thông báo tiếp theo, trước tiên, mô hình ngôn ngữ sẽ tính toán phân phối xác suất trên tất cả các mã thông báo trong từ vựng.

![example](https://huyenchip.com/assets/pics/sampling/1-sampling-tokens.png)

Đối với nhiệm vụ phân loại email spam, bạn có thể xuất giá trị có xác suất cao nhất. Nếu email có 90% khả năng là thư rác thì bạn phân loại email đó là thư rác. Tuy nhiên, đối với một mô hình ngôn ngữ, việc luôn chọn mã thông báo có khả năng xảy ra nhất, lấy mẫu tham lam sẽ tạo ra kết quả đầu ra nhàm chán. Hãy tưởng tượng một mô hình mà bất kỳ câu hỏi nào bạn hỏi đều luôn trả lời bằng những từ phổ biến nhất.

Thay vì luôn chọn mã thông báo có khả năng xảy ra tiếp theo nhất, chúng tôi có thể lấy mẫu mã thông báo tiếp theo theo phân bố xác suất trên tất cả các giá trị có thể có.


### Temperature

Một vấn đề khi lấy mẫu mã thông báo tiếp theo theo phân phối xác suất là mô hình có thể ít sáng tạo hơn. Trong ví dụ trước, các từ phổ biến chỉ các màu như đỏ, xanh lá cây, tím, v.v. có xác suất cao nhất. Câu trả lời của mô hình ngôn ngữ cuối cùng nghe giống như câu trả lời của một đứa trẻ 5 tuổi: Màu sắc yêu thích của tôi là màu xanh lá cây. Vì xác suất thấp nên mô hình ít có khả năng tạo ra câu sáng tạo như `Màu yêu thích của tôi là màu mặt hồ tĩnh lặng vào một buổi sáng mùa xuân`.

-> Temperature là một kỹ thuật được sử dụng để phân phối lại xác suất của các giá trị có thể. Theo trực quan, nó làm giảm xác suất của các mã thông báo phổ biến và kết quả là làm tăng xác suất của các mã thông báo hiếm hơn. Điều này cho phép các mô hình tạo ra nhiều phản hồi sáng tạo hơn.

Để hiểu nhiệt độ hoạt động như thế nào, hãy lùi lại một bước để xem mô hình tính toán xác suất như thế nào. Với một đầu vào, mạng nơ-ron xử lý đầu vào này và đưa ra một vectơ logit. Mỗi logit tương ứng với một logit có thể. Trong trường hợp mô hình ngôn ngữ, mỗi logit tương ứng với một mã thông báo trong từ vựng của mô hình. Kích thước vectơ logit là kích thước của từ vựng.

![example](https://huyenchip.com/assets/pics/sampling/2-logits.png)

Mặc dù **logit** lớn hơn tương ứng với xác suất cao hơn nhưng **logit** không đại diện cho xác suất.**Logit** không tổng hợp thành một. **Logit** thậm chí có thể âm, trong khi xác suất phải không âm. Để chuyển đổi **logit** thành xác suất, lớp ***softmax*** thường được sử dụng. Giả sử mô hình có vốn từ vựng là N và vectơ logit là **$[x_1,x_2,...,x_N]$**. Xác suất của lần thứ i $x_i$ là $p_i$ được tính theo công thức:
    $p_i = softmax(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$

Nhiệt độ là hằng số được sử dụng để điều chỉnh logit trước khi chuyển đổi softmax. Nhật ký được chia theo nhiệt độ.

Với một nhiệt độ nhất định là T, logit được điều chỉnh cho thứ i sex là $\frac{x_i}{T}$. Softmax sau đó được áp dụng trên logit đã điều chỉnh này thay vì trên $x_i$.

Hãy xem qua một ví dụ đơn giản để hiểu ảnh hưởng của nhiệt độ đến xác suất. Hãy tưởng tượng rằng chúng ta có một mô hình chỉ có hai đầu ra có thể có: A và B. Các bản ghi được tính toán từ lớp cuối cùng là [1, 3]. Logit của A là 1 và B là 3.

Không sử dụng nhiệt độ, tương đương với nhiệt độ = 1 thì xác suất softmax là [0,12, 0,88]. Mô hình chọn B 88% thời gian.

- Với nhiệt độ = 0,5 thì xác suất là [0,02, 0,98]. Mô hình chọn B 98% thời gian.

- Với nhiệt độ = 2 thì xác suất là [0,27, 0,73]. Mô hình chọn B 73% thời gian.

=> ***Nhiệt độ càng cao, mô hình càng ít có khả năng chọn giá trị rõ ràng nhất (giá trị có logit cao nhất), khiến kết quả đầu ra của mô hình trở nên sáng tạo hơn nhưng có khả năng kém mạch lạc hơn. Nhiệt độ càng thấp, mô hình càng có nhiều khả năng chọn giá trị rõ ràng nhất, làm cho mô hình trở nên nhất quán hơn nhưng có khả năng nhàm chán hơn.***

Biểu đồ bên dưới hiển thị xác suất softmax cho mã thông báo B ở các nhiệt độ khác nhau. Khi nhiệt độ tiến gần đến 0, xác suất mô hình chọn mã thông báo B sẽ gần bằng 1. Trong ví dụ của chúng tôi, đối với nhiệt độ dưới 0,1, mô hình hầu như luôn xuất ra B. Các nhà cung cấp mô hình thường giới hạn nhiệt độ trong khoảng từ 0 đến 2. Nếu bạn sở hữu mô hình của mình, bạn có thể sử dụng bất kỳ nhiệt độ không âm nào. Nhiệt độ 0,7 thường được khuyến nghị cho các trường hợp sử dụng sáng tạo vì nó cân bằng giữa tính sáng tạo và tính quyết định, nhưng bạn nên thử nghiệm và tìm ra nhiệt độ phù hợp nhất với mình.

![example](https://huyenchip.com/assets/pics/sampling/3-temperature.png)

Thông thường, hãy đặt nhiệt độ về 0 để kết quả đầu ra của mô hình nhất quán hơn. Về mặt kỹ thuật, nhiệt độ không bao giờ có thể bằng 0 – logit không thể chia cho 0. Trong thực tế, khi chúng ta đặt nhiệt độ thành 0, mô hình chỉ chọn mã thông báo có giá trị có logit lớn nhất, ví dụ: thực hiện một   argmax mà không thực hiện điều chỉnh logit và tính toán softmax.

Một kỹ thuật gỡ lỗi phổ biến khi làm việc với mô hình AI là xem xét xác suất mà mô hình này tính toán cho các đầu vào nhất định. Ví dụ: nếu xác suất trông ngẫu nhiên thì mô hình chưa học được nhiều. OpenAI trả về xác suất do mô hình của họ tạo ra dưới dạng logprobs. Logprobs, viết tắt của xác suất nhật ký, là xác suất trong thang đo log. Thang log được ưu tiên hơn khi làm việc với xác suất của mạng thần kinh vì nó giúp giảm vấn đề tràn. Một mô hình ngôn ngữ có thể hoạt động với kích thước từ vựng là 100.000, điều đó có nghĩa là xác suất của nhiều mã thông báo có thể quá nhỏ để có thể được biểu diễn bằng máy. Các số nhỏ có thể được làm tròn xuống 0. Thang log giúp giảm thiểu vấn đề này.


![example](https://huyenchip.com/assets/pics/sampling/4-logprobs.png)


    
### Top-k

Top-k là một chiến lược lấy mẫu để giảm khối lượng công việc tính toán mà không phải hy sinh quá nhiều tính đa dạng của phản hồi của mô hình. Hãy nhớ lại rằng để tính toán phân bố xác suất trên tất cả các giá trị có thể, lớp softmax được sử dụng.Softmax yêu cầu hai lần chuyển qua tất cả các giá trị có thể: một để thực hiện tổng số mũ ${\sum_{j} e^{x_j}}$ và một để biểu diễn $\frac{e^{x_i}}{\sum_{j} e^{x_j}}$ cho mỗi giá trị. ***Đối với một mô hình ngôn ngữ có vốn từ vựng lớn, quá trình này tốn kém về mặt tính toán.***

Softmax yêu cầu hai lần vượt qua tất cả các giá trị có thể: một để thực hiện tính tổng theo cấp số nhân và một để biểu diễn hàm softmax cho mỗi giá trị xi. Đối với một mô hình ngôn ngữ có vốn từ vựng lớn, quá trình này tốn kém về mặt tính toán.

-> Để tránh vấn đề này, sau khi mô hình tính toán nhật ký, chúng tôi chọn k nhật ký hàng đầu và chỉ thực hiện softmax trên k nhật ký hàng đầu này. Tùy thuộc vào mức độ đa dạng mà bạn muốn ứng dụng của mình, k có thể nằm trong khoảng từ 50 đến 500, nhỏ hơn nhiều so với kích thước từ vựng của mô hình. Sau đó, mô hình sẽ lấy mẫu từ các giá trị hàng đầu này. Giá trị k nhỏ hơn làm cho văn bản dễ dự đoán hơn nhưng kém thú vị hơn vì mô hình bị giới hạn ở một tập hợp nhỏ hơn các từ có khả năng xảy ra.

### Top-p

Trong lấy mẫu top-k, số lượng giá trị được xem xét cố định là k. Tuy nhiên, con số này sẽ thay đổi tùy theo tình hình.Ví dụ, đưa ra lời nhắc `Do you like music? Answer with only yes or no.`số lượng giá trị được xem xét phải là hai: `yes` and `no`. Với câu hỏi `What's the meaning of life?`, số lượng giá trị được xem xét sẽ lớn hơn nhiều.

Top-p, còn được gọi là lấy mẫu hạt nhân, cho phép lựa chọn các giá trị linh hoạt hơn để lấy mẫu. Trong lấy mẫu top-p, mô hình tính tổng xác suất của các giá trị tiếp theo có khả năng xảy ra nhất theo thứ tự giảm dần và dừng khi tổng đạt đến p. Chỉ các giá trị trong xác suất tích lũy này mới được xem xét. Các giá trị phổ biến để lấy mẫu top-p (hạt nhân) trong các mô hình ngôn ngữ thường nằm trong khoảng từ 0,9 đến 0,95. Ví dụ: giá trị top-p là 0,9 có nghĩa là mô hình sẽ xem xét tập giá trị nhỏ nhất có xác suất tích lũy vượt quá 90%.

Giả sử xác suất của tất cả các mã thông báo như trong hình bên dưới.Nếu top_p = 90%, chỉ `yes` và `maybe` sẽ được xem xét. vì xác suất tích lũy của chúng lớn hơn 90%. Nếu top_p = 99%, thì `yes`, `maybe`, và `no` được xem xét.

![example](https://huyenchip.com/assets/pics/sampling/5-top-p.png)

Không giống như top-k, top-p không nhất thiết phải giảm tải tính toán softmax. Lợi ích của nó là vì nó chỉ tập trung vào tập hợp các giá trị phù hợp nhất cho từng bối cảnh, nên nó cho phép kết quả đầu ra phù hợp hơn với ngữ cảnh. Về lý thuyết, việc lấy mẫu top-p dường như không mang lại nhiều lợi ích. Tuy nhiên, trên thực tế, top-p đã được chứng minh là có tác dụng tốt, khiến mức độ phổ biến của nó ngày càng tăng.

### Stopping condition

Mô hình ngôn ngữ tự hồi quy tạo ra các chuỗi mã thông báo bằng cách tạo hết mã thông báo này đến mã thông báo khác. Một chuỗi đầu ra dài sẽ mất nhiều thời gian hơn, tốn nhiều chi phí tính toán (tiền bạc) hơn và đôi khi có thể gây khó chịu cho người dùng. Chúng ta có thể muốn đặt điều kiện để mô hình dừng chuỗi.

Một phương pháp dễ dàng là yêu cầu các mô hình ngừng tạo sau một số lượng mã thông báo cố định. Nhược điểm là đầu ra có khả năng bị cắt giữa câu. Một phương pháp khác là sử dụng mã thông báo dừng. Ví dụ: bạn có thể yêu cầu các mô hình ngừng tạo khi gặp “<EOS>”. Điều kiện dừng rất hữu ích để giảm độ trễ và chi phí.

## 2. Test Time Sampling

Một cách đơn giản để cải thiện hiệu suất của mô hình là tạo ra nhiều đầu ra và chọn đầu ra tốt nhất. Cách tiếp cận này được gọi là `test time sampling` hoặc `test time compute`. Tôi thấy `test time compute` khó hiểu vì nó có thể được hiểu là lượng tính toán cần thiết để chạy thử nghiệm.

Bạn có thể hiển thị cho người dùng nhiều kết quả đầu ra và để họ chọn kết quả phù hợp nhất với họ hoặc nghĩ ra phương pháp để chọn kết quả tốt nhất. Nếu bạn muốn các phản hồi của mô hình nhất quán, bạn muốn giữ cố định tất cả các biến lấy mẫu. Tuy nhiên, nếu bạn muốn tạo nhiều kết quả đầu ra và chọn kết quả đầu ra tốt nhất, bạn không muốn thay đổi các biến lấy mẫu của mình.

Một phương pháp lựa chọn là chọn đầu ra có xác suất cao nhất. Đầu ra của mô hình ngôn ngữ là một chuỗi các mã thông báo, mỗi mã thông báo có một xác suất được mô hình tính toán. Xác suất của một đầu ra là tích của xác suất của tất cả các mã thông báo trong đầu ra.

Hãy xem xét chuỗi các mã thông báo [I, love, food] và:

xác suất để có I là 0,2

xác suất để được love and I là 0,1

xác suất để có được food chọn I and love là 0,3

Xác suất của chuỗi khi đó là: 0,2 * 0,1 * 0,3 = 0,006

Về mặt toán học, điều này có thể được biểu thị như sau:
```
    p(I love food)=p(I)×p(love|I)×p(food|I, love)
```

Hãy nhớ rằng việc xử lý các xác suất trên thang đo log sẽ dễ dàng hơn. Logarit của tích bằng tổng logarit, do đó logprob của một chuỗi mã thông báo là tổng logprob của tất cả các mã thông báo trong chuỗi.

```
logprob(I love food)=logprob(I)+logprob(love|I)+logprob(food|I, love)
```

Với tính tổng, các chuỗi dài hơn có thể phải làm giảm tổng logprob (log(1) = 0 và log của tất cả các giá trị dương nhỏ hơn 1 là âm). Để tránh thiên về các chuỗi ngắn, chúng tôi sử dụng logprob trung bình bằng cách chia tổng cho độ dài chuỗi của nó. Sau khi lấy mẫu nhiều đầu ra, chúng tôi chọn đầu ra có logprob trung bình cao nhất. Khi viết, đây là những gì OpenAI API sử dụng. Bạn có thể đặt tham số best_of thành một giá trị cụ thể, chẳng hạn như 10, để yêu cầu các mô hình OpenAI trả về kết quả có logprob trung bình cao nhất trong số 10 kết quả đầu ra khác nhau.

Một phương pháp khác là sử dụng mô hình khen thưởng để chấm điểm từng đầu ra, như đã thảo luận ở phần trước. Hãy nhớ lại rằng cả Stitch Fix và Grab đều chọn kết quả đầu ra được các mô hình phần thưởng hoặc người xác minh của họ cho điểm cao. OpenAI cũng đào tạo những người xác minh để giúp các mô hình của họ chọn ra giải pháp tốt nhất cho các bài toán (Cobbe và cộng sự, 2021). Họ phát hiện ra rằng việc lấy mẫu nhiều đầu ra hơn sẽ mang lại hiệu suất tốt hơn, nhưng chỉ ở một mức nhất định. Trong thí nghiệm của họ, điểm này là 400 kết quả đầu ra. Ngoài thời điểm này, hiệu suất bắt đầu giảm, như minh họa bên dưới. Họ đưa ra giả thuyết rằng khi số lượng kết quả đầu ra được lấy mẫu tăng lên thì khả năng tìm thấy những kết quả đầu ra đối nghịch có thể đánh lừa người xác minh cũng tăng lên. Mặc dù đây là một thử nghiệm thú vị nhưng tôi không tin bất kỳ ai trong quá trình sản xuất lấy mẫu 400 kết quả đầu ra khác nhau cho mỗi đầu vào. Chi phí sẽ rất lớn.

![example](https://huyenchip.com/assets/pics/sampling/6-test-time-sampling.png)

Bạn cũng có thể chọn phương pháp phỏng đoán dựa trên nhu cầu của ứng dụng của mình. Ví dụ: nếu ứng dụng của bạn được hưởng lợi từ phản hồi ngắn hơn, bạn có thể chọn phản hồi ngắn nhất. Nếu ứng dụng của bạn muốn chuyển đổi từ ngôn ngữ tự nhiên sang truy vấn SQL, bạn có thể chọn truy vấn SQL hợp lệ hiệu quả nhất.

Lấy mẫu nhiều kết quả đầu ra có thể hữu ích cho các nhiệm vụ mong đợi câu trả lời chính xác. Ví dụ: cho một bài toán, mô hình có thể giải bài toán đó nhiều lần và chọn câu trả lời thường gặp nhất làm giải pháp cuối cùng. Tương tự, đối với câu hỏi trắc nghiệm, mô hình có thể chọn tùy chọn đầu ra thường xuyên nhất. Đây là những gì Google đã làm khi đánh giá mô hình Gemini của họ trên MMLU, một chuẩn mực cho các câu hỏi trắc nghiệm. Họ đã lấy mẫu 32 kết quả đầu ra cho mỗi câu hỏi. Mặc dù điều này giúp Gemini đạt được điểm cao trong tiêu chuẩn này nhưng vẫn chưa rõ liệu mô hình của họ có tốt hơn mô hình khác có điểm thấp hơn khi chỉ tạo một đầu ra cho mỗi câu hỏi hay không.

Mô hình càng hay thay đổi thì chúng ta càng có thể hưởng lợi nhiều hơn từ việc lấy mẫu nhiều đầu ra. Tuy nhiên, điều tối ưu để làm với một mô hình hay thay đổi là hoán đổi nó bằng một mô hình khác. Đối với một dự án, chúng tôi đã sử dụng AI để trích xuất một số thông tin nhất định từ hình ảnh của sản phẩm. Chúng tôi nhận thấy rằng đối với cùng một hình ảnh, mô hình của chúng tôi chỉ có thể đọc được thông tin trong một nửa thời gian. Đối với nửa còn lại, người mẫu cho rằng hình ảnh quá mờ hoặc văn bản quá nhỏ để đọc. Đối với mỗi hình ảnh, chúng tôi đã truy vấn mô hình nhiều nhất ba lần cho đến khi nó có thể trích xuất thông tin.

Mặc dù chúng ta thường có thể mong đợi một số cải thiện hiệu suất của mô hình bằng cách lấy mẫu nhiều đầu ra, nhưng việc này rất tốn kém. Trung bình, việc tạo ra hai đầu ra có chi phí gần gấp đôi so với việc tạo ra một đầu ra.

## 3. Structured Outputs

Thông thường, trong quá trình sản xuất, chúng ta cần các mô hình để tạo văn bản theo các định dạng nhất định. Việc có kết quả đầu ra có cấu trúc là điều cần thiết cho hai tình huống sau.

- Các nhiệm vụ mà kết quả đầu ra cần phải tuân theo ngữ pháp nhất định. Ví dụ: đối với chuyển văn bản sang SQL hoặc chuyển văn bản sang biểu thức chính quy, kết quả đầu ra phải là các truy vấn và biểu thức chính quy SQL hợp lệ. Để phân loại, đầu ra phải là các lớp hợp lệ.
- Các tác vụ có kết quả đầu ra sau đó được phân tích cú pháp bởi các ứng dụng xuôi dòng. Ví dụ: nếu bạn sử dụng mô hình AI để viết mô tả sản phẩm, bạn chỉ muốn trích xuất các mô tả sản phẩm mà không có văn bản đệm như “Đây là mô tả” hoặc “Là mô hình ngôn ngữ, tôi không thể…”. Lý tưởng nhất là đối với trường hợp này, các mô hình nên tạo kết quả đầu ra có cấu trúc, chẳng hạn như JSON với các khóa cụ thể, có thể phân tích cú pháp được.

OpenAI là nhà cung cấp mô hình đầu tiên giới thiệu chế độ JSON trong API tạo văn bản của họ. Lưu ý rằng chế độ JSON của họ chỉ đảm bảo rằng kết quả đầu ra là JSON hợp lệ chứ không phải nội dung bên trong JSON. Khi viết bài, chế độ JSON của OpenAI vẫn chưa hoạt động đối với các mô hình thị giác, nhưng tôi chắc chắn rằng đó chỉ là vấn đề thời gian.

Các JSON được tạo cũng có thể bị cắt bớt do điều kiện dừng của mô hình, chẳng hạn như khi nó đạt đến độ dài mã thông báo đầu ra tối đa. Nếu độ dài mã thông báo tối đa được đặt quá ngắn thì JSON đầu ra có thể bị cắt ngắn và do đó không thể phân tích cú pháp được. Nếu nó được đặt quá lâu, phản hồi của mô hình sẽ trở nên quá chậm và tốn kém.

Các công cụ độc lập như hướng dẫn và phác thảo cho phép bạn cấu trúc kết quả đầu ra của một số mô hình nhất định. Dưới đây là hai ví dụ về cách sử dụng hướng dẫn để tạo kết quả đầu ra được giới hạn trong một tập hợp các tùy chọn và biểu thức chính quy.

![example](https://huyenchip.com/assets/pics/sampling/7-guidance.png)

### How to generate structured outputs

Bạn có thể hướng dẫn mô hình tạo các kết quả đầu ra có giới hạn ở các lớp khác nhau của ngăn xếp AI: trong quá trình nhắc, lấy mẫu và tinh chỉnh. `Prompting` hiện là phương pháp dễ nhất nhưng kém hiệu quả nhất. Bạn có thể hướng dẫn một mô hình xuất JSON hợp lệ theo một lược đồ cụ thể. Tuy nhiên, không có gì đảm bảo rằng mô hình sẽ luôn tuân theo hướng dẫn này.

Tinh chỉnh hiện là phương pháp phù hợp để giúp mô hình tạo kết quả đầu ra theo kiểu và định dạng mà bạn muốn. Bạn có thể thực hiện tinh chỉnh có hoặc không thay đổi kiến ​​trúc của mô hình. Ví dụ: bạn có thể tinh chỉnh mô hình trên các ví dụ với định dạng đầu ra mà bạn muốn. Mặc dù điều này vẫn không đảm bảo mô hình sẽ luôn xuất ra định dạng mong đợi nhưng điều này đáng tin cậy hơn nhiều so với lời nhắc. Nó cũng có thêm lợi ích là giảm chi phí suy luận, giả sử rằng bạn không còn phải đưa các hướng dẫn và ví dụ về định dạng mong muốn vào lời nhắc của mình nữa.

Đối với một số tác vụ nhất định, bạn có thể đảm bảo định dạng đầu ra bằng cách tinh chỉnh bằng cách sửa đổi kiến ​​trúc của mô hình. Ví dụ: để phân loại, bạn có thể thêm đầu phân loại vào kiến ​​trúc của mô hình nền tảng để đảm bảo rằng mô hình chỉ xuất ra một trong các lớp được chỉ định trước. Trong quá trình hoàn thiện, bạn có thể đào tạo lại toàn bộ kiến ​​trúc hoặc chỉ phần đầu phân loại này.

![example](https://huyenchip.com/assets/pics/sampling/8-finetuning-classifier.png)

Cả hai kỹ thuật lấy mẫu và tinh chỉnh đều cần thiết vì giả định rằng bản thân mô hình không có khả năng thực hiện được điều đó. Khi các mô hình trở nên mạnh mẽ hơn, chúng ta có thể kỳ vọng chúng sẽ làm theo hướng dẫn tốt hơn. Tôi nghi ngờ rằng trong tương lai, sẽ dễ dàng hơn để các mô hình xuất ra chính xác những gì chúng ta cần với sự nhắc nhở tối thiểu và những kỹ thuật này sẽ trở nên ít quan trọng hơn.

### Constraint sampling

Lấy mẫu ràng buộc là một kỹ thuật được sử dụng để hướng dẫn việc tạo văn bản theo các ràng buộc nhất định. Cách đơn giản nhất nhưng tốn kém nhất là tiếp tục tạo kết quả đầu ra cho đến khi bạn tìm thấy kết quả đầu ra phù hợp với ràng buộc của mình, như đã thảo luận trong phần Lấy mẫu thời gian thử nghiệm.

Việc lấy mẫu ràng buộc cũng có thể được thực hiện trong quá trình lấy mẫu mã thông báo. Tôi không thể tìm thấy nhiều tài liệu về cách các công ty ngày nay đang thực hiện điều đó. Những gì được viết dưới đây là theo hiểu biết của tôi, có thể sai, vì vậy rất mong nhận được phản hồi và gợi ý!

Ở cấp độ cao, để tạo mã thông báo, mô hình sẽ lấy mẫu giữa các giá trị đáp ứng các ràng buộc. Hãy nhớ lại rằng để tạo mã thông báo, trước tiên mô hình của bạn sẽ xuất ra một vectơ logit, mỗi logit tương ứng với một giá trị có thể có. Với việc lấy mẫu bị ràng buộc, chúng tôi lọc vectơ logit này để chỉ giữ lại các giá trị đáp ứng các ràng buộc của chúng tôi. Sau đó, chúng tôi lấy mẫu từ các giá trị hợp lệ này.

![example](https://huyenchip.com/assets/pics/sampling/9-constrained-sampling.png)

Trong ví dụ trên, ràng buộc rất dễ lọc. Tuy nhiên, trong hầu hết các trường hợp, nó không đơn giản như vậy. Chúng ta cần có một ngữ pháp quy định rõ điều gì được phép và không được phép ở mỗi bước. Ví dụ: ngữ pháp JSON quy định rằng sau {, chúng ta không thể có { khác trừ khi nó là một phần của chuỗi, như trong {"key": ""}.

Việc xây dựng ngữ pháp đó và kết hợp ngữ pháp đó vào quá trình lấy mẫu là điều không hề đơn giản. Chúng tôi cần một ngữ pháp riêng cho mọi định dạng đầu ra mà chúng tôi muốn: JSON, Regex, CSV, v.v. Một số phản đối việc lấy mẫu bị ràng buộc vì họ tin rằng các tài nguyên cần thiết cho việc lấy mẫu bị ràng buộc được đầu tư tốt hơn vào các mô hình đào tạo để làm theo hướng dẫn tốt hơn.








