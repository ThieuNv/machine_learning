## SONIA 
### Mạng cải thiện BP-NN. Gồm 3 tầng input, hidden, output 
- Mạng này quan trọng nhất là cái tầng hidden được tạo ra ngay trong quá trình training, số lượng hidden
unit tùy thuộc vào dữ liệu chứ không phải định nghĩa sẵn (active learning).
- Tầng hidden sử dụng giải thuật của hệ miễn dịch để tạo các hiddent unit.
- Các bước:

* B1: Khởi tạo dữ liệu training, validation, testing 
* B2: Đi tìm bộ trọng số giữa input layer và hidden layer. (Có 2 pha ở đây)
    *  Pha 1: Tạo cluster data áp dụng giải thuật hệ miễn dịch  
    *  Pha 2: Thêm artificial training data vào network sử dụng đột biến của hệ miễn dịch 
* B3: Đi tìm bộ trọng số giữa hidden layer và output layer 
    *  Ban đầu khởi tạo bằng 0
    *  Lúc update thì dùng backprogation (đạo hàm dựa vào hàm MSE) 
    
* B4: Test với tập trọng số của mạng 

### Chú ý:
- Ta có thể chỉ sử dụng pha 1 của B2 nhưng như vậy nó mất đi tính hiệu quả của đột biến trong hệ 
miễn dịch khi áp dụng vào mạng này.
- Pha 2 của B2 thêm vào để đối phó với testing data. 

- Quá trình update tập trọng số cũng khác bình thường. 

* B1: Khởi tạo bộ 1 (bộ giữa input và hidden)
* B2: Đối với từng training data (từng example). Nó sẽ làm như sau :
    *  2.1- Tính khoảng cách giữa input và hidden (Nói chung là tìm center cho input, hiểu đây đang làm là phân cụm các input vào các hidden unit) 
        *   Nếu có thì update weight bộ 1, sau đó tiếp tục phân cho đến khi phân hết các input 
        *   Nếu không có thì tạo 1 node hidden unit mới (Đây là lí do vì sao ta không kiểm soát số lượng hidden unit ngay từ đầu, việc nó tạo ra hidden unit mới có thể hiểu đối với example này 
        nó đang lưu trữ local data cần thiết sẽ dùng về sau, hoặc có thể hiểu là nó đang làm cho việc phân bố dữ liệu trong input space tốt hơn)
    *  2.2- Sau khi nó đã tìm cho tất cả các input được hidden unit gần nhất, nó sẽ kiểm tra xem các hidden unit đã cover toàn bộ input space chưa, 
    nếu chưa thì bắt đầu thực hiện pha 2 bước 2 bên trên. (Có thể đoạn này sẽ tạo ra thêm nhiều mutated hidden unit nữa)
    *  2.3- Sau khi kết thúc pha 2, nó sẽ tính giá trị đầu ra của các hidden unit. 
    *  2.4- Sau đó khởi tạo bộ 2 (bộ giữa hidden và output)
    *  2.5- Tiếp theo tính output và tính lỗi MSE. Sau đó tính các đạo hàm của MSQ (đang dùng backpropagation) với w và bias của hidden layer.
    *  2.6- Update bộ 2
    
    *  2.7- Tiếp tục với training data tiếp theo 
    
==> Nhận xét: 
+ Hidden unit được tạo dựa vào giải thuật hệ miễn dịch (tính khoảng cách - phân cụm, đột biến - local data memory) 
+ Hidden unit được tạo trong quá trình training và không được định nghĩa trước.
+ Training kiểu SGD với từng example. 
+ Vẫn sử dụng backprogation để update bộ trọng số giữa hidden và output.
+ Không sử dụng các khái niệm như epoch, batch_size. 
+ Có 2 tham số learning-rate-h sử dụng update weight pha 1 và positive number (0, 1) sử dụng update weight cho pha 2.
+ Các tham số quan trọng là: stimulation-level (Khoảng cách tối thiểu) cho pha 1 và distance level, threshold number of input vector associated with.


## LINK 
https://stackoverflow.com/questions/40034993/how-to-get-element-wise-matrix-multiplication-hadamard-product-in-numpy
https://www.mathworks.com/help/nnet/ref/tansig.html
https://stackoverflow.com/questions/6088077/how-to-get-a-random-number-between-a-float-range
https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.zeros.html
https://www.programiz.com/python-programming/methods/list/sort
https://ai.stackexchange.com/questions/3428/mutation-and-crossover-in-a-genetic-algorithm-with-real-numbers

## Link paper 
https://drive.google.com/drive/folders/1z_VnYfD-Eh7LQE-eGk8gofnQUFQB0Ufg?usp=sharing    
    
    
