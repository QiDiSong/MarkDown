#include <iostream>
#include <vector>
using namespace std;
#include <list>
int main()
{
    vector<int> dim = {1, 2, 3, 4, 5}; // 定义一个整数向量dim，并初始化为{1, 2, 3, 4, 5}
    for (auto i : dim)
        cout << i << " ";
    cout << endl;

    vector<int> lst = {1, 2, 3, 4, 5, 6}; // 定义一个整数列表lst，并初始化为{1, 2, 3, 4, 5}
    vector<int> dim2(lst.begin(), lst.end()); // 使用列表lst的begin()和end()迭代器，将列表中的元素复制到向量dim2中
    for (auto i : dim2)
        cout << i << " ";

    return 0;
}