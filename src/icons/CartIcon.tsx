import { ShoppingCart as LucideShoppingCart, LucideProps } from 'lucide-react';

const CartIcon = ({ className, ...props }: LucideProps) => {
  return <LucideShoppingCart className={className} {...props} />;
};

export default CartIcon;