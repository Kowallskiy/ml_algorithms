#pragma warning(disable: 5054) // Disable warning C5054
#pragma warning(disable: 4127)

#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <set>

int main() {
    int x, y, z;
    std::cin >> x >> y >> z;
    int n;

    std::cin >> n;

    std::string s = std::to_string(n);

    int res = 0;

    std::set<int> set;

    for (char c : s) {
        int i = c;
        set.insert(i);
    }

    for (int a : set) {
        std::cout << a << "\n";
        if (a != x && a != y && a != z) {
            --res;
        }
    }

    res = res + static_cast<int>(set.size());
    std::cout << res << "\n";

    return 0;
}
