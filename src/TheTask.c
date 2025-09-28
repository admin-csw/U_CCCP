#include <stdio.h>
#include "TheEmployeesSalary.h"

void TaskDoer(const double TheSalaries[], double TheSumArray[], int SIZE) {
    for (int i = 0; i < SIZE; i++) {
        TheSumArray[i] = TheSalaries[i] + (TheSalaries[i] * 15/100) + 5000;
    }
}

int main() {
    int size = sizeof(TheArrayOfSalaries) / sizeof(TheArrayOfSalaries[0]); // 100
    double TheNewSalaries[size];
    TaskDoer(TheArrayOfSalaries, TheNewSalaries, size);
    for (int i = 0; i < size; i++) {
        printf("Old Salary: %.2f, New Salary: %.2f\n", TheArrayOfSalaries[i], TheNewSalaries[i]);
    }

    return 0;
}
