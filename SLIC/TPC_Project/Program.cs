using System;

namespace TPC_Project
{
    class Program
    {
        static void Main(string[] args)
        {
            bool exit = false;

            while (!exit)
            {
                Console.Clear();
                Console.ForegroundColor = ConsoleColor.Cyan;
                Console.WriteLine("=============================================================");
                Console.WriteLine("  PARALLEL IMAGE SEGMENTATION USING K-MEANS FOR BRAIN TUMOR  ");
                Console.WriteLine("=============================================================");
                Console.ResetColor();

                Console.WriteLine("\nMain Menu:");
                Console.WriteLine("1. Say Hello");
                Console.WriteLine("2. Add Two Numbers");
                Console.WriteLine("3. Show Current Date & Time");
                Console.WriteLine("4. Exit");

                Console.Write("\nEnter your choice: ");

                var choice = Console.ReadLine();

                switch (choice)
                {
                    case "1":
                        SayHello();
                        break;
                    case "2":
                        AddNumbers();
                        break;
                    case "3":
                        ShowDateTime();
                        break;
                    case "4":
                        exit = true;
                        Console.WriteLine("Goodbye!");
                        break;
                    default:
                        Console.WriteLine("Invalid choice. Please enter 1, 2, 3 or 4.");
                        break;
                }
                Console.WriteLine("\nPress Enter to continue...");
                Console.ReadLine();
            }
        }

        // Function 1: Greet the user
        static void SayHello()
        {
            Console.Write("Enter your name: ");
            var name = Console.ReadLine();
            Console.WriteLine($"Hello, {name}! Welcome to the program.");
        }

        // Function 2: Add two numbers entered by the user
        static void AddNumbers()
        {
            Console.Write("Enter first number: ");
            if (!double.TryParse(Console.ReadLine(), out double num1))
            {
                Console.WriteLine("Invalid number.");
                return;
            }

            Console.Write("Enter second number: ");
            if (!double.TryParse(Console.ReadLine(), out double num2))
            {
                Console.WriteLine("Invalid number.");
                return;
            }

            double sum = num1 + num2;
            Console.WriteLine($"The sum of {num1} and {num2} is {sum}.");
        }

        // Function 3: Display current date and time
        static void ShowDateTime()
        {
            DateTime now = DateTime.Now;
            Console.WriteLine($"Current date and time: {now}");
        }
    }
}