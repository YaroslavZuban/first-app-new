package com.eltex;

import java.util.Scanner;

public class Profit {
    public static void main(String[] args) {
        final var scanner = new Scanner(System.in);

        System.out.println("Введите сумму покупок за прошлый год");

        final var clientYearlyPurchases = scanner.nextInt();
        final var discount = 0.02;
        final var discountStart = 3_000;

        final double totalDiscount;

        if (clientYearlyPurchases >= discountStart) {
            totalDiscount = clientYearlyPurchases * discount;
        } else {
            totalDiscount = 0;
        }

        if (totalDiscount > 0) {
            System.out.printf("Попробуйте нашу новую подписку и сэкономьте 2%%, ваша экономия составит %s ₽%n", totalDiscount);
        } else {
            System.out.println("За прошлый год вы бы сэкономили с подпиской 0 ₽");
        }
    }
}