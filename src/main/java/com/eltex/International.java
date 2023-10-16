package com.eltex;

import java.util.Scanner;

public class International {
    public static void main(String[] args) {
        System.out.println("Введите валюту: 1 – RUB, 2 – BYN, 3 – KZT");

        final var scanner = new Scanner(System.in);

        var numberCurrency = scanner.nextInt();
        var currencySymbol = "RUB";

        while (true) {
            if (numberCurrency == 1) {
                currencySymbol = "RUB";
                break;
            } else if (numberCurrency == 2) {
                currencySymbol = "BYN";
                break;
            } else if (numberCurrency == 3) {
                currencySymbol = "KZT";
                break;
            } else {
                System.out.println("Не было найдено такой валюты. Попробуйте заново.");
                System.out.println();
                System.out.println("Введите валюту: 1 – RUB, 2 – BYN, 3 – KZT");

                numberCurrency = scanner.nextInt();
            }
        }

        System.out.println("Введите сумму покупок за прошлый год");

        final var clientYearlyPurchases = scanner.nextInt();
        final var discount = 0.02;
        final var discountStart = 3_000;

        final double totalDiscount;

        if (clientYearlyPurchases < discountStart) {
            totalDiscount = 0;
        } else {
            totalDiscount = clientYearlyPurchases * discount;
            System.out.printf("За прошлый год вы бы сэкономили с подпиской %s %s", totalDiscount, currencySymbol);
            System.out.println();
        }

        if (totalDiscount > 0) {
            System.out.printf("Попробуйте нашу новую подписку и сэкономьте 2%%, ваша экономия составит %s ₽%n", totalDiscount);
        } else {
            System.out.println("За прошлый год вы бы сэкономили с подпиской 0 ₽");
        }
    }
}
