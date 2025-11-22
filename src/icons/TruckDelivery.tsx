import { Truck as LucideTruck, LucideProps } from 'lucide-react';

const TruckDelivery = ({ className, ...props }: LucideProps) => {
  return <LucideTruck className={className} {...props} />;
};

export default TruckDelivery;