#include <stdio.h>

int main(void){


    // char c = 'v';
    // int *p = &c;
    // printf("Value:  %i\n", *(p - 3));
    int arr[] = {1, 2, 3, 4, 5};     
    int length = sizeof(arr)/sizeof(arr[0]);    
    
    printf("[");
    for (int i = 0, length = sizeof(arr)/sizeof(arr[0]); i < length; i++) {     
        printf("%d ", arr[i]);     
    }      
    printf("\b]");
    printf("\n");

    // int x = 65;
    // char *ch = &x;
    // printf("Value:  %c\n", *ch);


}